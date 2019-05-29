import numpy as np

# data_1 = np.load('train_data_1.npy')
# data_2 = np.load('train_data_2.npy')

label_1 = np.load('train_label.npy')
# label_2 = np.load('train_label_2.npy')

# label_1 = np.squeeze(label_1, axis =1 )

# label_1 = np.expand_dims(label_1, axis = 0)
print(label_1)

print(label_1.shape)

# print(label_2.shape)
# train_data = np.append(data_1, data_2, axis = 0)

# train_label = np.append(label_1, label_2, axis = 0)

# print(train_data.shape)
# print(train_label.shape)

# train_data = np.expand_dims(train_data, axis = 1)
# print(train_data.shape)

# np.save('train_data', train_data)
# np.save('train_label', label_1)