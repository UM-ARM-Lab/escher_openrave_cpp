import math
import numpy as np

# com_dict
# x
# 0: (-Inf, -0.3)
# 1: [-0.3, -0.2)
# 2: [-0.2, -0.1)
# 3: [-0.1, 0.0)
# 4: [0.0, 0.1)
# 5: [0.1, 0.2)
# 6: [0.2, Inf)
# y
# 0: (-Inf, -0.1)
# 1: [-0.1, 0.0)
# 2: [0.0, 0.1)
# 3: [0.1, 0.2)
# 4: [0.2, Inf)
# z
# 0: (-Inf, 0.8)
# 1: [0.8, 0.9)
# 2: [0.9, 1.0)
# 3: [1.0, 1.1)
# 4: [1.1, Inf)
def com_index(x):
    idxx = max(min(int(math.floor(x[-6] * 10) + 4), 6), 0)
    idxy = max(min(int(math.floor(x[-5] * 10) + 2), 4), 0)
    idxz = max(min(int(math.floor(x[-4] * 10) - 7), 4), 0)
    return (idxx, idxy, idxz)

def main():
    x = np.zeros((24,), dtype=float)
    x[-4] = 0.63
    print(com_index(x))
    x[-4] = 0.73
    print(com_index(x))
    x[-4] = 0.83
    print(com_index(x))
    x[-4] = 0.93
    print(com_index(x))
    x[-4] = 1.03
    print(com_index(x))
    x[-4] = 1.13
    print(com_index(x))
    x[-4] = 1.23
    print(com_index(x))


                

if __name__ == '__main__':
    main()