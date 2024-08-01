#!/usr/bin/env python

# CuPy demonstration https://cupy.dev/
#
# Complementary code for The Quest for Performance Part II : Perl vs Python.
# https://chrisarg.github.io/Killing-It-with-PERL/
#
# Author: Mario Roy, August 1, 2024

import argparse
import cupy as cp
import numpy as np
import time

kernel = cp.RawKernel(r"""
extern "C" __global__ void compute_inplace(float *array, int m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    array[i] = sqrt(array[i]);
    array[i] = sin(array[i]);
    array[i] = cos(array[i]);
}
""", "compute_inplace")

def divide_up(dividend, divisor):
    """
    Helper funtion to get the next up value for integer division.
    """
    return dividend // divisor + 1 if dividend % divisor else dividend // divisor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    parser.add_argument("--incl-synctime", default=False, action='store_true')
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    # Making a copy is from learning how-to using the framework, optional
    array0 = cp.random.rand(args.arraysize, dtype=cp.float32)
    arrayb = cp.array(array0)

    block_size = 32
    num_blocks = divide_up(args.arraysize, block_size)

    for _ in range(10):
        start_time = time.time()
        kernel((num_blocks,), (block_size,), (arrayb, args.arraysize))
        if args.incl_synctime:
            cp.cuda.Device().synchronize()
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python CuPy - GPU CUDA)")
    print("=" * 50)

    try:
        main()
    except KeyboardInterrupt:
        pass

