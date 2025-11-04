import argparse
import numpy as np
import sys
import math

from cerebras.sdk import sdk_utils # type: ignore # pylint: disable=no-name-in-module

star_coefficients = [
  np.array([0.25,0.25,0.0,0.25,0.25], dtype=np.float32),
  np.array([0.0625, 0.0625, 0.0625, 0.0625, 0.0, 0.0625, 0.0625, 0.0625, 0.0625], dtype=np.float32),
  np.array([0.0625,0.0625,0.125,0.0625,0.0625,0.125,0.0,0.125,0.0625,0.0625,0.125,0.0625,0.0625], dtype=np.float32)
]

box_coefficients = [
  np.array([0.125,  0.125,  0.125,
            0.125,    0.0,  0.125,
            0.125,  0.125,  0.125], dtype=np.float32),
  np.array([0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            0.03125,  0.0625,  0.0625,  0.0625, 0.03125,
            0.03125,  0.0625,     0.0,  0.0625, 0.03125,
            0.03125,  0.0625,  0.0625,  0.0625, 0.03125,
            0.03125, 0.03125, 0.03125, 0.03125, 0.03125,], dtype=np.float32),
  np.array([0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0625, 0.0625, 0.0625, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0625,    0.0, 0.0625, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0625, 0.0625, 0.0625, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
            0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,], dtype=np.float32)
]

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', help="the test compile output dir")
  parser.add_argument('--arch', help="the simulation target architecture")
  parser.add_argument('--cmaddr', help="IP:port for CS system")
  parser.add_argument("--verify", action="store_true", help="Verify Y computation")

  args = parser.parse_args()
  verify = args.verify

  return args, verify

def get_coefficients(shape, radius):

  # default stencil kernel
  if(shape == "star2d"):
    return star_coefficients[radius-1]
  elif(shape == "box2d"):
    return box_coefficients[radius-1]
  else:
    raise Exception(f'Shape "{shape}" does not exist!')

def parse_tsc(w, h, tsc):
  min_cycles = math.inf
  max_cycles = 0
  min_w = 0
  max_w = 0

  for x in range(w):
    for y in range(h):
      cycles = sdk_utils.calculate_cycles(tsc[x, y, :])

      if cycles < min_cycles:
        min_cycles = cycles
        min_w = x
        min_h = y
      if cycles > max_cycles:
        max_cycles = cycles
        max_w = x
        max_h = y

  return {"min": min_cycles, "max": max_cycles, "min_pe":(min_w,min_h), "max_pe":(max_w, max_h)}

def generate_input(M, N, shape="random", value=10):

  if(shape == "random"):
    np.random.seed(42)
    A = (np.random.rand(M,N) * 5).astype(np.float32)
  elif(shape == "index"):
    A = np.reshape([i for i in range(0,M*N)], (M,N)).astype(np.float32)
  elif(shape == "diagonal"):
    A = diagonal_input(M, N, value)

  return A
  
def diagonal_input(M, N, value):

  A = np.zeros((M,N))

  for i in range(M):
    A[i,i] = value

  return A.astype(np.float32)

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


def check_result(A, result, M, N, c, shape, radius, iterations):
  print("Checking Result")

  expected = cpu_stencil(A.copy(), M, N, c, shape, radius, iterations)

  np.testing.assert_allclose(result, expected, atol=0, rtol=0)
  print("SUCCESS!\n")

'''
  Computes the stencil in the same order as wse kernel
  - center, north, south, west, east
  - starting from the outer halo
'''
def cpu_stencil(A, m, n, c, shape, radius=1, iters=1):

  if(shape == "star2d"):
    y = star_stencil(A, m, n, c, radius, iters)
  elif(shape == "box2d"):
    y = box_stencil(A, m, n, c, radius, iters)
  else:
    raise Exception(f'Shape "{shape}" does not exist!')

  return y  

def star_stencil(A, m, n, c, radius, iters):
  y = A.ravel()
  y_aux = np.zeros(m*n, dtype=np.float32)

  for _ in range(iters):
    for i in range(m):
      for j in range(n):
        idx = i*n+j

        if(i == j): 
          y_aux[idx] = y[idx]
        else:
          y_aux[idx] = c[2*radius] * y[idx] # center

          for r in range(radius): # north
            if(i-(radius-r) >= 0): y_aux[idx] += c[r] * y[(i-(radius-r))*n+j]

          for r in range(radius): # south
            if(i+(radius-r) < m): y_aux[idx] += c[len(c)-r-1] * y[(i+(radius-r))*n+j]

          for r in range(radius): # west
            if(j-(radius-r) >= 0): y_aux[idx] += c[radius+r] * y[i*n+j-(radius-r)]

          for r in range(radius): # east
            if(j+(radius-r) < n): y_aux[idx] += c[2*radius + r + 1] * y[i*n+j+(radius-r)]

        

    y, y_aux = y_aux, y
  
  return y

def box_stencil(A, m, n, c, radius, iters):
  y = A.ravel()
  y_aux = np.zeros(m*n, dtype=np.float32)

  s = np.sqrt(len(c)).astype(np.int32) # stencil side (e.g. 25 -> 5x5)

  for _ in range(iters):
    for i in range(m):
      for j in range(n):
        idx = i*n+j

        if(i == j): 
          y_aux[idx] = y[idx]
        else:
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
