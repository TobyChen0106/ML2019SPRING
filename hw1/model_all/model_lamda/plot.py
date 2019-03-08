import numpy as np
import matplotlib.pyplot as plt



x1 = np.load('loss2epo_lamda=0.1.npy')
x2 = np.load('loss2epo_lamda=0.01.npy')
x3 = np.load('loss2epo_lamda=0.001.npy')
x4 = np.load('loss2epo_lamda=0.0001.npy')

it = range(1000, 100000, 100)

plt.plot(it, x1[10:], label='lamda = 0.1')
plt.plot(it, x2[10:], label='lamda = 0.01')
plt.plot(it, x3[10:], label='lamda = 0.001')
plt.plot(it, x4[10:], label='lamda = 0.0001')

plt.legend(loc='upper right')

# plt.yscale('iteration')
# plt.title('Loss to Iteration')
# plt.yscale('logit')
plt.grid(True)


plt.show()
