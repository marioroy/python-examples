#!/usr/bin/env python

# NumPy demonstration https://numpy.org/
#
# Complementary code for The Quest for Performance Part II : Perl vs Python.
# https://chrisarg.github.io/Killing-It-with-PERL/
#
# Author: Mario Roy, August 1, 2024

import argparse
import numpy as np
import time

def compute_inplace(array):
    np.sqrt(array, array)
    np.sin(array, array)
    np.cos(array, array)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    # Making a copy is unnecessary for this demo, simply exploring
    array0 = np.random.rand(args.arraysize).astype(np.float32)
    arrayb = np.array(array0)

    for _ in range(10):
        start_time = time.time()
        compute_inplace(arrayb)
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python NumPy - CPU single-threaded)")
    print("=" * 50)

    try:
        main()
    except KeyboardInterrupt:
        pass

