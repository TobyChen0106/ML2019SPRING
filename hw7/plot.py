import matplotlib.pyplot as plt
import numpy as np

v_tsne = np.load('data/visualizationx_tsne.npy')
print('v_tsne shape', v_tsne.shape)

x1 = v_tsne[:2500,0]
y1 = v_tsne[:2500,1]

x2 = v_tsne[2500:,0]
y2 = v_tsne[2500:,1]

plt.scatter(x1, y1,  marker='o', c='green', label='data[0:2500]')
plt.scatter(x2, y2,  marker='o', c='orange', label='data[2500:5000]')
plt.legend(loc='upper right')
# Show the boundary between the regions:
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

plt.show()


    
