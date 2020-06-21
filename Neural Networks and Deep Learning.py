import numpy as np
a = np.random.rand(1000000,1)
b = np.random.rand(1000000,1)
c = np.array([1, 2, 3])
d = np.array([1, 2, 3])

print(np.dot(a.T, b))
sol=0
for i in range(1000000):
    sol+= a[i]*b[i]
print(sol)
a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = a*b
print(c)