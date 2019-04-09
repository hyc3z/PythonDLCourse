import matplotlib.pyplot as plt
import math
import numpy as np

x = np.linspace(0.0001, 0.9999, 1000)
# y = [0, 1]
y = np.linspace(0, 1, 5)
for j in y:
    plt.plot(x, [-j * math.log(i) - (1 - j) * math.log(1 - i) for i in x], label="y="+str(j))
plt.legend()
plt.show()

