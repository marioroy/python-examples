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

def compute_inplace(array):
    cp.sqrt(array, array)
    cp.sin(array, array)
    cp.cos(array, array)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    parser.add_argument("--incl-synctime", default=False, action='store_true')
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    # Making a copy is from learning how-to using the framework, optional
    array0 = cp.random.rand(args.arraysize, dtype=cp.float32)
    arrayb = cp.array(array0)

    for _ in range(10):
        start_time = time.time()
        compute_inplace(arrayb)
        if args.incl_synctime:
            cp.cuda.Device().synchronize()
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python CuPy - GPU cuBLAS)")
    print("=" * 50)

    try:
        main()
    except KeyboardInterrupt:
        pass

