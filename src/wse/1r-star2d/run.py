#!/usr/bin/env cs_python

import json
import numpy as np
import time

from utils import *

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # type: ignore # pylint: disable=no-name-in-module

# Read arguments
args, verify = read_args()

# Get parameters from compile metadata
with open(f"{args.name}/out.json", "r", encoding="utf8") as f:
    data = json.load(f)

w = int(data['params']['kernel_dim_x'])
h = int(data['params']['kernel_dim_y'])
N = int(data['params']['N'])
M = int(data['params']['M'])
iterations = int(data['params']['iterations'])
radius = 1

# Input
heat_value = 10
A = generate_input(M, N, "diagonal", value=heat_value)
A_prepared = prepare_input(A, M, N, h, w, radius)

coefficients = get_coefficients("star2d", radius)
c_tiled = np.tile(coefficients, w*h)

pad_x, pad_y = 0, 0
if (M % h != 0): pad_x = h - (M%h)
if (N % w != 0): pad_y = w - (N%w)
pe_M = (M + pad_x) // h
pe_N = (N + pad_y) // w
tiled_shape = (pe_M + 2*radius, pe_N + 2*radius)
elements_per_PE = (pe_M + 2*radius) * (pe_N + 2*radius)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

runner.load()
runner.run()

A_symbol = runner.get_id('A')
coeff_symbol = runner.get_id('c')
symbol_maxmin_time = runner.get_id("maxmin_time")

# Load matrix
A_prepared = prepare_input(A, M, N, h, w, radius)

start_time = time.perf_counter()

runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Load coefficients
runner.memcpy_h2d(coeff_symbol, c_tiled, 0, 0, w, h, 5, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch program
runner.launch('compute', nonblock=False)

start_time_compute = time.perf_counter()

# Retrieve result
y_result = np.zeros(elements_per_PE*h*w, dtype=np.float32)
runner.memcpy_d2h(y_result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

end_time_compute = time.perf_counter()

# Retrieve timings
tsc = np.zeros((w*h*3), dtype=np.uint32)
runner.memcpy_d2h(tsc, symbol_maxmin_time, 0, 0, w, h, 3,
  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

end_time = time.perf_counter()

runner.stop()

####################
##  Check Result  ##
####################

if verify:

  y_result = y_result.reshape(w, h, *tiled_shape)
  y_result = y_result[:,:, radius:-radius, radius:-radius].transpose(0, 2, 1, 3)
  y_result = y_result.reshape(M+pad_x, N+pad_y)[:M,:N].ravel()

  check_result(A, y_result, M, N, coefficients, "star2d", radius, iterations)


###################
##  Timestamps   ##
###################

cycles = parse_tsc(w, h, tsc.view(np.float32).reshape((h, w, 3)))
time_device = cycles["max"] / (875e6)
GStencil = (M * N * iterations) / time_device * 10e-9

time_h2d = start_time_compute - start_time
time_compute = end_time_compute - start_time_compute
time_d2h = end_time - end_time_compute
time_total = end_time - start_time

print(f'Time (device): {time_device} s')
print(f'GStencil/s: {GStencil}')
print(f'{w},{h},{M},{N},{iterations},{time_h2d},{time_compute},{time_d2h},{time_total},{(M * N * iterations) / time_compute * 10e-9}')

with open("star2d-1r.csv", "a") as f:
  f.write(f'{w},{h},{M},{N},{iterations},{time_h2d},{time_compute},{time_d2h},{time_total},{GStencil}\n')