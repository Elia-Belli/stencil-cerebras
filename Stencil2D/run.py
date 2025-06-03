#!/usr/bin/env cs_python

import argparse
import json
import numpy as np
import math

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # type: ignore # pylint: disable=no-name-in-module
from cerebras.sdk import sdk_utils # type: ignore # pylint: disable=no-name-in-module


# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
parser.add_argument("--verify", action="store_true", help="Verify Y computation")
args = parser.parse_args()

verify = args.verify

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
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True, simfab_numthreads=8)

# Get symbols
A_symbol = runner.get_id('A')
symbol_maxmin_time = runner.get_id("maxmin_time")

# Load and run the program
runner.load()
runner.run()

A_prepared = A.reshape(w, M_per_PE, h, N_per_PE).transpose(0, 2, 1, 3)
A_prepared = np.pad(A_prepared, ((0, 0), (0, 0), (halo, halo), (halo, halo)), mode='constant', constant_values=0).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
print("Copying A onto Device...")

runner.launch('compute', nonblock=False)
print("DONE!")
print("Starting Computation...")

if verify:
  runner.memcpy_d2h(y_result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
  print("DONE!")
  print("Copying back result...")

  y_result = y_result.reshape(w, h, *tiled_shape)
  y_result = y_result[:,:, halo:-halo, halo:-halo].transpose(0, 2, 1, 3)


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

print("Cycle Counts:")
print("Min cycles (", min_w, ", ", min_h, "): ", min_cycles)
print("Max cycles (", max_w, ", ", max_h, "): ", max_cycles)


flops = (M*N) * 5 * 2 * iterations  # 5 fmac (2 flops) per f32 at each iteration
cells = (M*N) * iterations
time = max_cycles / 850e6 # cycles / clock freq in (s)

print("\nTotal FLOPs: ", flops)
print("Total Cells: ", cells)
print(f'GFlop/s: {flops / time * 10e-9:.2e} (TOTAL)' )
print(f'GStencil/s: {cells / time * 10e-9} (TOTAL)')

tile_flops = flops/(h*w)
tile_cells = cells/(h*w)
print("\nFLOPs per Tile: ", tile_flops)
print("Cells per Tile: ", tile_cells)
print(f'GFlop/s: {tile_flops / time * 10e-9:.2e}')
print(f'GStencil/s: {tile_cells / time * 10e-9}')

