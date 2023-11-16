import numpy as np
import matplotlib.pyplot as plt

with open('./out/ANN_loss.bin', 'rb') as f:
    data = np.fromfile(f)

data = data.reshape(-1, 2)

plt.plot(data[:, 0], label='Validation Loss')
plt.plot(data[:, 1], label='Training Loss')
plt.legend()
plt.savefig('./out/loss.png')