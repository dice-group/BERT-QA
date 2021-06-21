import matplotlib.pyplot as plt
import numpy as np
import math

import matplotlib.pyplot as plt
x=np.arange(-10,10,0.1)


def relu(x):
    return np.maximum(0, x)


y=relu(x)
plt.plot(x,y)
plt.show()

