import numpy as np

# 假设objectPixelRow和objectPixelCol是已经定义好的数组
objectPixelRow = [1, 2, 3, 4, 5]
objectPixelCol = [6, 7, 8, 9, 10]

objectPixels = np.column_stack((objectPixelRow, objectPixelCol))

print((objectPixels))