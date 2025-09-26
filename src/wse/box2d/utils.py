import argparse
import numpy as np
import sys

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help="the test compile output dir")
  parser.add_argument('--arch', help="the simulation target architecture")
  parser.add_argument('--stencil', nargs='*', type=int, help="the stencil kernel")  
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

'''
  Computes the stencil in the same order as wse kernel
  - center, north, south, west, east
  - starting from the outer halo
'''
def cpu_stencil(A, m, n, c, radius=1, iters=1):
  y = A.ravel()
  y_aux = np.zeros(m*n, dtype=np.float32)

  s = np.sqrt(len(c)).astype(np.int32) # stencil side (e.g. 25 -> 5x5)

  for _ in range(iters):
    for i in range(m):
      for j in range(n):
        idx = i*n+j

        y_aux[idx] = c[radius*(s+1)] * y[idx]  # C

        for r in range(0, radius):
          if(i-(radius-r) >= 0): y_aux[idx] += c[radius + (r)*s] * y[(i-(radius-r))*n+j]  # N
        
        for r in range(0, radius):
          if(i+(radius-r) <  m): y_aux[idx] += c[radius + (s-1-r)*s] * y[(i+(radius-r))*n+j]  # S

        for r in range(0, radius):
          if(j-(radius-r) >= 0): y_aux[idx] += c[(radius*s) + r] * y[i*n+ j-(radius-r)]  # W

        for r in range(0, radius):
          if(j+(radius-r) <  n): y_aux[idx] += c[(radius+1)*s - (r+1)] * y[i*n+ j+(radius-r)]  # E

        for ri in range(0, radius):
          for rj in range(0, radius):
            if(i-(radius-ri) >=0 and j-(radius-rj) >=0):
              y_aux[idx] += c[ri * s + rj] * y[(i-(radius-ri))*n + j-(radius-rj)]  # NW
        
        for ri in range(0, radius):
          for rj in range(0, radius):
            if(i-(radius-ri) >=0 and j+(radius-rj) < n):
              y_aux[idx] += c[ri * s + (s -rj-1)] * y[(i-(radius-ri))*n + j+(radius-rj)]  # NE

        for ri in range(0, radius):
          for rj in range(0, radius):
            if(i+(radius-ri) < m and j-(radius-rj) >= 0):
              y_aux[idx] += c[(s-1 - ri)*s + rj] * y[(i+(radius-ri))*n + j-(radius-rj)]  # SW

        for ri in range(0, radius):
          for rj in range(0, radius):
            if(i+(ri+1) < m and j+(rj+1) < n):
              y_aux[idx] += c[(radius+1 + ri)*s + (radius+1 +rj)] * y[(i+1+ri)*n + j+1+rj]  # SE

    y, y_aux = y_aux, y

  return y  


def get_coefficients(stencil, radius):

  n_weights = (radius*2 + 1) * (radius*2+1)

  if (len(stencil) == n_weights):
    return np.array(stencil, dtype=np.float32)
  elif  (radius < 4):
    print(f'Input stencil and radius do not match, using default radius {radius} weights')
  else:
    sys.exit('For radius > 4 insert your own stencil kernel!')

  if (radius == 1):
      return np.array([ 0.125,  0.125,  0.125,
                        0.125,     -1,  0.125,
                        0.125,  0.125,  0.125], dtype=np.float32)
  elif (radius == 2):
      return np.array([ 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
                        0.03125,  0.0625,  0.0625,  0.0625, 0.03125,
                        0.03125,  0.0625,      -1,  0.0625, 0.03125,
                        0.03125,  0.0625,  0.0625,  0.0625, 0.03125,
                        0.03125, 0.03125, 0.03125, 0.03125, 0.03125,], dtype=np.float32)
  elif (radius == 3):
      return np.array([ 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0625, 0.0625, 0.0625, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0625,     -1, 0.0625, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0625, 0.0625, 0.0625, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
                        0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,], dtype=np.float32)
                        

  