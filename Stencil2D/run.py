#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

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
A = np.arange(M*N, dtype=np.float32)

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

# Load and run the program
runner.load()
runner.run()

# Prepare A
A_prepared = A.reshape(w, M_per_PE, h, N_per_PE).transpose(0, 2, 1, 3)
A_prepared = np.pad(A_prepared, ((0, 0), (0, 0), (halo, halo), (halo, halo)), mode='constant', constant_values=0).ravel()
# Copy A to device
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.launch('compute', nonblock=False)

# Copy y back from device
runner.memcpy_d2h(y_result, A_symbol, 0, 0, w, h, elements_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
# Unpack y_result
y_result = y_result.reshape(w, h, *tiled_shape)
y_result = y_result[:,:, halo:-halo, halo:-halo].transpose(0, 2, 1, 3)

# Stop the program
runner.stop()


# Ensure that the result matches our expectation
y = A.copy()
y_aux = np.zeros(M*N, dtype=np.float32)

for _ in range(iterations):
  for i in range(M):
    for j in range(N):
      if(i-1 >= 0): y_aux[i*N+j] += y[(i-1)*N+j]
      if(i+1 < M):  y_aux[i*N+j] += y[(i+1)*N+j]
      if(j-1 >= 0): y_aux[i*N+j] += y[i*N+j-1]
      if(j+1 < N):  y_aux[i*N+j] += y[i*N+j+1]
      y_aux[i*N+j] -= 4*y[i*N+j]

  temp = y.copy()
  y = y_aux.copy()
  y_aux = temp.copy()

y_expected = y.copy()    

print(f'{A.reshape(M,N)}\n')
print(f'{y_expected.reshape(M,N).astype(np.int32)}\n')  #remove cast to int32
print(f'{y_result.reshape(M,N).astype(np.int32)}\n')    #remove cast to int32
np.testing.assert_allclose(y_result.ravel(), y_expected, atol=0.01, rtol=0)
print("SUCCESS!")
