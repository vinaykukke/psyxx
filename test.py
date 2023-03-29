# import Numpy
import numpy as np

array = np.eye(4, k=1)
# array[array == 0] = 2
# array[:2] = 9
# array_two = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# array_two[:, ::2] = 9
# array[array < 5] = 3
# array[2:] = 90
# array[:] = 3.14
# print(array)
array_two = array.reshape(-1)
print(array_two)
