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
from numba import njit, prange, set_parallel_chunksize

@njit('void(f4[:])', nogil=True, parallel=True)
def compute_inplace(array):
    m = array.shape[0]
  # set_parallel_chunksize(10000)
    for i in prange(m):
        array[i] = math.cos(math.sin(math.sqrt(array[i])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    array0 = np.random.rand(args.arraysize).astype(np.float32)
    array_copy = np.array(array0)

    for _ in range(10):
        start_time = time.time()
        compute_inplace(array_copy)
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python Numba - many CPU threads)")
    print("=" * 50)

    try:
        main()
    except KeyboardInterrupt:
        pass

