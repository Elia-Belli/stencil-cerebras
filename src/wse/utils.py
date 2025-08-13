import argparse
import numpy as np

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help="the test compile output dir")
  parser.add_argument('--arch', help="the simulation target architecture")
  parser.add_argument('--cmaddr', help="IP:port for CS system")
  parser.add_argument("--verify", action="store_true", help="Verify Y computation")
  parser.add_argument("--verbose", action="store_true", help="Print computation details")
  parser.add_argument("--traces", action="store_true", help="Capture Fabric Traces")

  args = parser.parse_args()
  verify = args.verify
  verbose = args.verbose
  suppress_traces = not args.traces

  return args, verify, verbose, suppress_traces


def prepare_input(input, input_m, input_n, fabric_x, fabric_y, halo):

  pad_x, pad_y = 0, 0
  if (input_m % fabric_y != 0): pad_x = fabric_y - (input_m % fabric_y)
  if (input_n % fabric_x != 0): pad_y = fabric_x - (input_n % fabric_x)

  A_padded = np.pad(input, [(0,pad_x), (0,pad_y)], mode="constant").ravel()

  pe_M = (input_m + pad_x) // fabric_y
  pe_N = (input_n + pad_y) // fabric_x

  A_reshaped = A_padded.reshape(fabric_x, pe_M, fabric_y, pe_N).transpose(0, 2, 1, 3)
  A_prepared = np.pad(A_reshaped, ((0, 0), (0, 0), (halo, halo), (halo, halo)), mode='constant', constant_values=0).ravel()

  return A_prepared

def cpu_stencil(A, m, n, iters):
  y = A.ravel()
  y_aux = np.zeros(m*n, dtype=np.float32)

  for _ in range(iters):
    for i in range(m):
      for j in range(n):
        if(i-1 >= 0): y_aux[i*n+j] += y[(i-1)*n+j]
        if(i+1 < m) : y_aux[i*n+j] += y[(i+1)*n+j]
        if(j-1 >= 0): y_aux[i*n+j] += y[i*n+j-1]
        y_aux[i*n+j] -= 4.0*y[i*n+j]
        if(j+1 < n) : y_aux[i*n+j] += y[i*n+j+1]

    y, y_aux = y_aux, y

  return y  