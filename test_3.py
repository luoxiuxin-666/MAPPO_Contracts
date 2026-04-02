import random
import numpy as np
rng = np.random.default_rng(seed=42)
range1 = [10,20]

a = rng.integers(range1[0], range1[1])
d = rng.integers(range1[0], range1[1])



class UAV:
    def __init__(self,f_k,d_k,c_k):
        self.f_k = f_k
        self.d_k = d_k
        self.c_k = c_k

    def out_info(self):
        print(f"the f_k is {self.f_k},and the d_k is {self.d_k},and the c_k is {self.c_k}")

UAVs = []
for i in range(10):
    f_k = rng.integers(range1[0], range1[1])
    d_k = rng.integers(range1[0], range1[1])
    c_k = np.round(rng.uniform(range1[0], range1[1]),2)
    UAVs.append(UAV(f_k,d_k,c_k))

for i in range(10):
    UAVs[i].out_info()
