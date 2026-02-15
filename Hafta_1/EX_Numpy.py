import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 2, 3], [4, 5, 6]])

print("x:\n{}".format(x))
print("------*20------")

y = np.linspace(-10, 10, 100)
z = np.sin(y)

plt.plot(y, z, color="y", marker="o") 

plt.show()