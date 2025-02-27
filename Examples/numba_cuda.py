#!/usr/bin/env python

# Numba demonstration https://numba.pydata.org/
#
# Complementary code for The Quest for Performance Part II : Perl vs Python.
# https://chrisarg.github.io/Killing-It-with-PERL/
#
# Author: Mario Roy, August 1, 2024

import argparse
import numpy as np
import math
import time
from numba import cuda

@cuda.jit('void(f4[:])')
def compute_inplace(array):
    i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    if i >= array.shape[0]: return
    array[i] = math.cos(math.sin(math.sqrt(array[i])))

def divide_up(dividend, divisor):
    """
    Helper funtion to get the next up value for integer division.
    """
    return dividend // divisor + 1 if dividend % divisor else dividend // divisor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    parser.add_argument("--incl-asnumpy", default=False, action='store_true')
    parser.add_argument("--incl-synctime", default=False, action='store_true')
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    array0 = np.random.rand(args.arraysize).astype(np.float32)
    arrayb = cuda.to_device(array0)
    cuda.synchronize()

    block_size = 32
    num_blocks = divide_up(args.arraysize, block_size)
    array_np = np.zeros(args.arraysize, dtype=np.float32)

    for _ in range(10):
        start_time = time.time()
        compute_inplace[num_blocks, block_size](arrayb)
        if args.incl_asnumpy:
            arrayb.copy_to_host(array_np)
        if args.incl_synctime:
            cuda.synchronize()
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python Numba - GPU CUDA)")
    print("=" * 50)

    try:
        main()
    except KeyboardInterrupt:
        pass

