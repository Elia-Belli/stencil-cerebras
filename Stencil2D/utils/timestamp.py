import struct
import numpy as np

def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

