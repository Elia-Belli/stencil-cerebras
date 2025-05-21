#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
import math

from utils.timestamp import float_to_hex, sub_ts

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # type: ignore # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

w = int(compile_data['params']['kernel_dim_x'])
h = int(compile_data['params']['kernel_dim_y'])

N = int(compile_data['params']['N'])
M = int(compile_data['params']['M'])

iterations = int(compile_data['params']['iterations'])
print(f"\nSimulation Info\n  rows: {M}\n  cols: {N}\n  iterations: {iterations}\
      \n\nFabric Info\n  width: {w}\n  height: {h}\n")

# Construct A
np.random.seed(0)
A = (np.random.rand(N*M) * 10).astype(np.float32)

M_per_PE = M // h
N_per_PE = N // w
halo = 1
tiled_shape = (M_per_PE + 2*halo, N_per_PE + 2*halo)
elements_per_PE = (M_per_PE + 2*halo) * (N_per_PE + 2*halo)

y_result = np.zeros(elements_per_PE*h*w, dtype=np.float32)

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A on device
A_symbol = runner.get_id('A')
symbol_maxmin_time = runner.get_id("maxmin_time")
# Load and run the program
runner.load()
runner.run()

print("Copying A onto Device...")
A_prepared = A.reshape(w, M_per_PE, h, N_per_PE).transpose(0, 2, 1, 3)
A_prepared = np.pad(A_prepared, ((0, 0), (0, 0), (halo, halo), (halo, halo)), mode='constant', constant_values=0).ravel()
print("DONE!")
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

print("Starting Computation...")
runner.launch('compute', nonblock=False)
print("DONE!")


print("Copying back result...")
# Copy y back from device
runner.memcpy_d2h(y_result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# Unpack y_result
y_result = y_result.reshape(w, h, *tiled_shape)
y_result = y_result[:,:, halo:-halo, halo:-halo].transpose(0, 2, 1, 3)
print("DONE!")


print("Copying back timestamps")
data = np.zeros((w*h*3), dtype=np.uint32)
runner.memcpy_d2h(data, symbol_maxmin_time, 0, 0, w, h, 3,
  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
maxmin_time_hwl = data.view(np.float32).reshape((h, w, 3))
print("DONE!\n")


# Stop the program
runner.stop()


############################
##      Check Result      ##
############################
print("Verifying result\n")
y = np.zeros(M*N, dtype=np.float32)
temp = np.zeros(M*N, dtype=np.float32)
y_aux = np.zeros(M*N, dtype=np.float32)
y_expected = np.zeros(M*N, dtype=np.float32)

y = A.copy()

# calculating host result
for _ in range(iterations):
  for i in range(M):
    for j in range(N):
      if(i-1 >= 0): y_aux[i*N+j] += y[(i-1)*N+j]
      if(i+1 < M) : y_aux[i*N+j] += y[(i+1)*N+j]
      if(j-1 >= 0): y_aux[i*N+j] += y[i*N+j-1]
      y_aux[i*N+j] -= 4.0*y[i*N+j]
      if(j+1 < N) : y_aux[i*N+j] += y[i*N+j+1]


  temp = y.copy()
  y = y_aux.copy()
  y_aux = temp.copy()

y_expected = y.copy()    

print(f'Input:\n{A.reshape(M,N)}\n')
#print(f'{y_expected.reshape(M,N)}\n')  
print(f'Output:\n{y_result.reshape(M,N)}\n')  

with open("./out/output/wse_result.txt", "w+") as f:
  for i in y_result.ravel():
    print(f'{i}', file=f)  

with open("./out/output/py_result.txt", "w+") as f:
  for i in y_expected:
    print(f'{i}', file=f)  

np.testing.assert_allclose(y_result.ravel(), y_expected, atol=0, rtol=0)
print("SUCCESS!\n")


###################
##  Timestamps   ##
###################
tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
min_cycles = math.inf
max_cycles = 0

for w in range(w):
  for h in range(h):
    hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 0)]), base=16)
    hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 1)]), base=16)
    hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
    tsc_tensor_d2h[0] = hex_t0 & 0x0000ffff
    tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000ffff
    tsc_tensor_d2h[2] = hex_t1 & 0x0000ffff
    tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000ffff
    tsc_tensor_d2h[4] = hex_t2 & 0x0000ffff
    tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000ffff

    cycles = sub_ts(tsc_tensor_d2h)
    if cycles < min_cycles:
      min_cycles = cycles
      min_w = w
      min_h = h
    if cycles > max_cycles:
      max_cycles = cycles
      max_w = w
      max_h = h

print("Cycle Counts:")
print("Min cycles (", min_w, ", ", min_h, "): ", min_cycles)
print("Max cycles (", max_w, ", ", max_h, "): ", max_cycles)