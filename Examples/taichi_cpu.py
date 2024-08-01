#!/usr/bin/env python

# Taichi Lang demonstration https://www.taichi-lang.org/
#
# Complementary code for The Quest for Performance Part II : Perl vs Python.
# https://chrisarg.github.io/Killing-It-with-PERL/
#
# I tried using external arrays as Taichi kernel arguments while learning.
# It performs well on arch ti.cpu only.
#
# Author: Mario Roy, August 1, 2024

import argparse
import taichi as ti
import numpy as np
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=os.cpu_count())
parser.add_argument("--arraysize", type=int, default=100_000_000)
args = parser.parse_args()

# If the kernel is run on the CPU backend, N threads will be used
num_threads = int(max(1, args.workers))

@ti.kernel
def compute_inplace(array: ti.types.ndarray()):
    m = array.shape[0]
    ti.loop_config(parallelize=num_threads)
    for i in range(m):
        array[i] = ti.cos(ti.sin(ti.sqrt(array[i])))

def main():
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
    print("In place in (Python Taichi - CPU multi-threaded)")
    print("=" * 50)

    ti.init(arch=ti.cpu)
    try:
        main()
    except KeyboardInterrupt:
        pass

