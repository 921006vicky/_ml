import random

def hillClimbing3D(f, x, y, z, h=0.01):
    while True:
        fxyz = f(x, y, z)
        print('x={0:.3f} y={1:.3f} z={2:.3f} f(x,y,z)={3:.3f}'.format(x, y, z, fxyz))
        # 找鄰近點中函數值更小的方向
        if f(x + h, y, z) < fxyz:
            x += h
        elif f(x - h, y, z) < fxyz:
            x -= h
        elif f(x, y + h, z) < fxyz:
            y += h
        elif f(x, y - h, z) < fxyz:
            y -= h
        elif f(x, y, z + h) < fxyz:
            z += h
        elif f(x, y, z - h) < fxyz:
            z -= h
        else:
            break
    return (x, y, z, fxyz)

# 目標函數
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 起始點設為 (0,0,0)
hillClimbing3D(f, 0, 0, 0)
