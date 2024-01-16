import numpy as np

_y = slice(3, 6, 2)

myslice = (Ellipsis, _y)


x = np.random.rand(100, 100)

x[myslice]

x[:, 3]

dx = 30
dy = -10
slice = (
    slice(0+dx, -1, 1) if dx > 0 else slice(0, -dx, 1),
    slice(0+dy, -1, 1) if dx > 0 else slice(0, -dy, 1)
)



crop = x[slice]
print(crop.shape)


