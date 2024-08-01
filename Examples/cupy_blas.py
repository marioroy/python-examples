
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
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    array0 = cp.random.rand(args.arraysize, dtype=cp.float32)
    array_copy = cp.array(array0)

    for _ in range(10):
        start_time = time.time()
        compute_inplace(array_copy)
      # cp.cuda.Device().synchronize()
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python CuPy - cuBLAS)")
    print("=" * 50)

    main()

