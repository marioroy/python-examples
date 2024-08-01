#!/usr/bin/env python

# Taichi Lang demonstration https://www.taichi-lang.org/
#
# Complementary code for The Quest for Performance Part II : Perl vs Python.
# https://chrisarg.github.io/Killing-It-with-PERL/
#
# This variant runs well on arch ti.cpu, ti.gpu, ti.opengl, and ti.vulkan.
# Not tested ti.metal.
#
# Author: Mario Roy, August 1, 2024

import argparse
import taichi as ti
import numpy as np
import time

@ti.kernel
def init(array: ti.template()):
    for i in array:
        array[i] = ti.random(ti.float32)

@ti.kernel
def compute_inplace(array: ti.template()):
    for i in array:
        array[i] = ti.cos(ti.sin(ti.sqrt(array[i])))

@ti.kernel
def copy_to_numpy(np_arr: ti.types.ndarray(), src_arr: ti.template()):
    for i in src_arr:
        np_arr[i] = src_arr[i]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", type=int, default=100_000_000)
    parser.add_argument("--incl-asnumpy", default=False, action='store_true')
    parser.add_argument("--incl-synctime", default=False, action='store_true')
    args = parser.parse_args()

    # Generate the data structures for the benchmark
    # Making a copy is from learning how-to using the framework, optional
    array0 = ti.field(shape=args.arraysize, dtype=ti.float32)
    arrayb = ti.field(shape=args.arraysize, dtype=ti.float32)
    ti.sync()

    init(array0)
    arrayb.copy_from(array0)
    array_np = np.zeros(args.arraysize, dtype=np.float32)

    for _ in range(10):
        start_time = time.time()
        compute_inplace(arrayb)
        if args.incl_asnumpy:
            copy_to_numpy(array_np, arrayb)
        if args.incl_synctime:
            ti.sync()
        elapsed_time = time.time() - start_time
        print(f"{elapsed_time * 1e6:12.3f} µs")

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python Taichi - GPU auto)")
    print("=" * 50)

    ti.init(arch=ti.gpu) # auto-detect GPU backend
    try:
        main()
    except KeyboardInterrupt:
        pass

