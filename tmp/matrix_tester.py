import time, random,sys, gc
import numpy as np
from copy import deepcopy

gc.collect()

in_size = 100
out_size = 100
sparsity = 0.98

x = np.random.random(in_size)
A = np.random.random((out_size, in_size))

length_all = in_size*out_size
S = np.random.random(length_all)
print("1:", sys.getsizeof(S))
S[:int(length_all*sparsity)] = np.zeros(int(length_all*sparsity))
np.random.shuffle(S)
print("2:", sys.getsizeof(S))
S_2dim = S.reshape((out_size, in_size)).copy()
print("3:", sys.getsizeof(S), sys.getsizeof(S_2dim), S is S_2dim)
# print(S)
S_comp = []
for i in range(out_size):
    idx = np.where(abs(S_2dim[i]) > 0.)[0]
    # print(idx)
    if len(idx) != 0:
        S_comp.append((i, idx, S_2dim[i][idx]))

a1 = deepcopy(A.tolist())
a2 = deepcopy(S_2dim.tolist())
# print(a1)
# print(a2)
print(S_comp)

print(A.nbytes, S_2dim.nbytes, len(S_comp), sys.getsizeof(0.00001))
print("-"*20)

b = np.random.random(out_size)

y_sparse = np.zeros(out_size)

y_origin = np.matmul(x, A.T) + b

# print(x)
# print(A)
# print(b)
print("-"*100)
# print(y_origin)
