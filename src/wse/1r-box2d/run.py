#!/usr/bin/env cs_python

import json
import numpy as np
import math

from utils import *

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # type: ignore # pylint: disable=no-name-in-module
from cerebras.sdk import sdk_utils # type: ignore # pylint: disable=no-name-in-module

# Read arguments
args, verify, verbose, suppress_traces = read_args()

# Get parameters from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

w = int(compile_data['params']['kernel_dim_x'])
h = int(compile_data['params']['kernel_dim_y'])
N = int(compile_data['params']['N'])
M = int(compile_data['params']['M'])
iterations = int(compile_data['params']['iterations'])
halo = int(compile_data['params']['radius'])

# Input
np.random.seed(0)
A = (np.random.rand(M,N) * 10).astype(np.float32)
A = np.reshape([i for i in range(0,M*N)], (M,N)).astype(np.float32)

pad_x, pad_y = 0, 0
if (M % h != 0): pad_x = h - (M%h)
if (N % w != 0): pad_y = w - (N%w)
pe_M = (M + pad_x) // h
pe_N = (N + pad_y) // w
tiled_shape = (pe_M + 2*halo, pe_N + 2*halo)
elements_per_PE = (pe_M + 2*halo) * (pe_N + 2*halo)


# Stencil
coefficients = get_coefficients(args.stencil, halo)
# coefficients = np.array([ 0, 0, 0,
#                           0, 0, 0,
#                           0, 0, 1], dtype=np.float32)

c_tiled = np.tile(coefficients, w*h)

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
A_prepared = prepare_input(A, M, N, h, w, halo)
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
  y_result = np.zeros(elements_per_PE*h*w, dtype=np.float32)
  runner.memcpy_d2h(y_result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  print("DONE!")
  print("Copying back y_result...")

  y_result = y_result.reshape(w, h, *tiled_shape)
  y_result = y_result[:,:, halo:-halo, halo:-halo].transpose(0, 2, 1, 3)
  y_result = y_result.reshape(M+pad_x, N+pad_y)[:M,:N].ravel()

# Retrieve timings
data = np.zeros((w*h*3), dtype=np.uint32)
runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, w, h, 3,
  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
maxmin_time_hwl = data.view(np.float32).reshape((h, w, 3))
print("DONE!")
print("Copying back timestamps...")

# Stop the program
runner.stop()
print("DONE!\n")


# ############################
# ##      Check Result      ##
# ############################
if verify:
  print("Checking Result")

  y_expected = cpu_stencil(A.copy(), M, N, coefficients, halo, iterations)

  if(verbose):
    print(f'Input:\n{A.reshape(M,N)}\n')
    print(f'Expected:\n{y_expected.reshape(M,N)}\n')  
    print(f'Output:\n{y_result.reshape(M,N)}\n')  

  with open("../../../logs/wse_result.txt", "w+") as f:
    for i in y_result:
      print(f'{i}', file=f)  

  with open("../../../logs/cpu_result.txt", "w+") as f:
    for i in y_expected:
      print(f'{i}', file=f)  

  np.testing.assert_allclose(y_result, y_expected, atol=0, rtol=0)
  print("SUCCESS!\n")


###################
##  Timestamps   ##
###################
tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
min_cycles = math.inf
max_cycles = 0

for x in range(w):
  for y in range(h):
    cycles = sdk_utils.calculate_cycles(maxmin_time_hwl[x, y, :])

    if cycles < min_cycles:
      min_cycles = cycles
      min_w = x
      min_h = y
    if cycles > max_cycles:
      max_cycles = cycles
      max_w = x
      max_h = y

flops = (M*N) * len(coefficients) * 2 * iterations 
cells = (M*N) * iterations
time = max_cycles / 875e6 if (args.arch == "wse3") else 850e6
tile_cells = cells/(h*w)
GStencil = cells / time * 10e-9

if(verbose):
  print("Cycle Counts:")
  print("Min cycles (", min_w, ", ", min_h, "): ", min_cycles)
  print("Max cycles (", max_w, ", ", max_h, "): ", max_cycles)

  print("Total Cells: ", cells)
  print(f'GStencil/s: {cells / time * 10e-9} (TOTAL)')


# print y_results to file
with open("../../../logs/run_test_log.csv", "a") as f:
  print(f'{w},{h},{M},{N},{iterations},{GStencil},{min_cycles},{max_cycles},{args.arch},box2d-{halo}r (reusing send colors)', file=f)
