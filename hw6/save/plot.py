import numpy as np
import matplotlib.pyplot as plt

folder = 'RNN_WORD'
dnn_train_loss = np.load(folder+'/train_loss_log.npy')
dnn_train_acc = np.load(folder+'/train_acc_log.npy')
dnn_val_loss = np.load(folder+'/val_loss_log.npy')
dnn_val_acc = np.load(folder+'/val_acc_log.npy')



it = range(0, 20)

plt.figure(num='acc', figsize=(2, 1))
plt.subplot(2, 1, 1)
plt.plot(it, dnn_train_acc, label='train_acc')
plt.plot(it, dnn_val_acc, label='val_acc')
plt.legend(loc='upper right')

# plt.figure(num='loss', figsize=(1, 2))
plt.subplot(2, 1, 2)
plt.plot(it, dnn_train_loss, label='train_loss')
plt.plot(it, dnn_val_loss, label='val_loss')


plt.legend(loc='upper right')

# plt.yscale('iteration')
# plt.title('Loss to Iteration')
# plt.yscale('logit')
plt.grid(True)


plt.show()
