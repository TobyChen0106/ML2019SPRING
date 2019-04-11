import numpy as np
import matplotlib.pyplot as plt




dnn_train_loss = np.load('acc_loss_dnn/train_loss_log.npy')
dnn_train_acc = np.load('acc_loss_dnn/train_acc_log.npy')
dnn_val_loss = np.load('acc_loss_dnn/val_loss_log.npy')
dnn_val_acc = np.load('acc_loss_dnn/val_acc_log.npy')

cnn_train_loss = np.load('acc_loss_cnn/train_loss_log.npy')
cnn_train_acc = np.load('acc_loss_cnn/train_acc_log.npy')
cnn_val_loss = np.load('acc_loss_cnn/val_loss_log.npy')
cnn_val_acc = np.load('acc_loss_cnn/val_acc_log.npy')

it = range(0, 50)

plt.plot(it, dnn_train_loss, label='dnn_train_loss')
plt.plot(it, dnn_train_acc, label='dnn_train_acc')
# plt.plot(it, cnn_train_loss, label='cnn_train_loss')
# plt.plot(it, cnn_train_acc, label='cnn_train_acc')

plt.legend(loc='upper right')

# plt.yscale('iteration')
# plt.title('Loss to Iteration')
# plt.yscale('logit')
plt.grid(True)


plt.show()
