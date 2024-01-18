import numpy as np
import matplotlib.pyplot as plt

# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8, 6), dpi=80)
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y = np.cos(x)
plt.plot(x, y)

