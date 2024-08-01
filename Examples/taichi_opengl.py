#!/usr/bin/env python

from taichi_gpu import main
import taichi as ti

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python Taichi - OpenGL)")
    print("=" * 50)

    ti.init(arch=ti.opengl)
    try:
        main()
    except KeyboardInterrupt:
        pass

