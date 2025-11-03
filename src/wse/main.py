#!/usr/bin/env cs_python

import json
import numpy as np
import os

from utils import *
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # type: ignore # pylint: disable=no-name-in-module

# Read arguments
args, verify, suppress_traces = read_args()

# Get parameters from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

w = int(compile_data['params']['kernel_dim_x'])
h = int(compile_data['params']['kernel_dim_y'])
N = int(compile_data['params']['N'])
M = int(compile_data['params']['M'])
iterations = int(compile_data['params']['iterations'])
radius = int(compile_data['params']['radius'])

print(args.program)

# Input
A = generate_input(M, N, "random")
A_prepared = prepare_input(A, M, N, h, w, radius)

coefficients = get_coefficients(program, radius)
c_tiled = np.tile(coefficients, w*h)

# pad and split
pad_x, pad_y = 0, 0
if (M % h != 0): pad_x = h - (M%h)
if (N % w != 0): pad_y = w - (N%w)
pe_M = (M + pad_x) // h
pe_N = (N + pad_y) // w
tiled_shape = (pe_M + 2*radius, pe_N + 2*radius)
elements_per_PE = (pe_M + 2*radius) * (pe_N + 2*radius)

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr,
                    suppress_simfab_trace=suppress_traces, 
                    simfab_numthreads=8,
                    msg_level = "INFO")

# Get symbols
A_symbol = runner.get_id('A')
coeff_symbol = runner.get_id('c')
symbol_maxmin_time = runner.get_id("maxmin_time")

# Load and run the program
runner.load()
runner.run()

# Load matrix
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
print("\nCopying A onto Device...")

# Load coefficients
runner.memcpy_h2d(coeff_symbol, c_tiled, 0, 0, w, h, len(coefficients), streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
print("DONE!")
print("Loading coefficients...")

# Launch program
runner.launch('compute', nonblock=False)
print("DONE!")
print("Starting Computation...")

# Retrieve result for correctness check
if verify:
  result = np.zeros(elements_per_PE*h*w, dtype=np.float32)
  runner.memcpy_d2h(result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  print("DONE!")
  print("Copying back result...")

  result = result.reshape(w, h, *tiled_shape)
  result = result[:,:, radius:-radius, radius:-radius].transpose(0, 2, 1, 3)
  result = result.reshape(M+pad_x, N+pad_y)[:M,:N].ravel()

# Retrieve timings
data = np.zeros((w*h*3), dtype=np.uint32)
runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, w, h, 3,
  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print("DONE!")
print("Copying back timestamps...")

# Stop the program
runner.stop()
print("DONE!\n")


##########################
##  Checks and Metrics  ##
##########################
if (verify) : check_result(A, coefficients, result, M, N, radius, iterations)

cycles = parse_tsc(w, h, data.view(np.float32).reshape((h, w, 3)))
time = cycles["max"] / (875e6 if (args.arch == "wse3") else 850e6)
GStencil = (M * N * iterations) / time * 10e-9

print(f'Time: {time} s')
print(f'GStencil/s: {GStencil}')


real_sim = "sim" if(runner.is_simulation()) else "real"
with open(log, "a") as f:
  print(f'star2d-{radius}r,{args.arch}-{real_sim},{w},{h},{M},{N},{iterations},,{time},,{GStencil}', file=f)