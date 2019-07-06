import numpy as np
import matplotlib.pyplot as plt

input_path = 'toy_output.asc'
output_path = 'out.png'

pic = np.loadtxt(input_path, skiprows=6)
plt.imshow(pic)
plt.savefig(output_path)
