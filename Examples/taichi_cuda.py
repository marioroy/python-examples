
from taichi_gpu import main
import taichi as ti

if __name__ == "__main__":
    print("=" * 50)
    print("In place in (Python Taichi - CUDA)")
    print("=" * 50)

    ti.init(arch=ti.cuda)
    main()

