import numpy as np
import sys

batch_size = 3360 # 3360 = 84 * 40
epochs = 500000
lr = 0.1
lamb = 0.95

# raw data
def preprocess(path):
	n_features = 18
	data = np.genfromtxt(path, delimiter=',', encoding='latin1')[1:, 3:]
	
	# original delete 722 ~ 1441 (march and april)
	# data = np.delete(data, range(721,1441), axis=0)

	# data = np.delete(data, range(10,data.shape[0], n_features), axis=0) # delete RAINFALL (NR)
	for i in range(10, data.shape[0], n_features):
		# data[i] = np.zeros(24) # convert RAINFALL to 0
		for j in range(data[i].shape[0]): # convert nan(NR) to 0
			if np.isnan(data[i][j]):
				data[i][j] = 0
	n_data = data.shape[0] // 18

	# interpolation for negative value
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if data[i][j] < 0:
				if j == 0 or data[i][j-1]<0:
					if data[i][j+1] > 0:
						data[i][j] = data[i][1]
					else:
						data[i][j] = 0
				elif j == 23 or data[i][j+1]<0:
					if data[i][j-1] > 0:
						data[i][j] = data[i][22]
					else:
						data[i][j] = 0
				else:
					data[i][j] = (data[i][j-1] + data[i][j+1]) / 2

	# contact data into (12,18,480)
	# concat_data = []
	# data = np.array_split(data, n_data)

	# for i in range(12):
	# 	concat_data.append( np.concatenate(data[(i*20):((i+1)*20)], axis=1) )
	# concat_data = np.array(concat_data)

	# data.shape = (240*18,24)
	# data indexed 9 is PM2.5
	x_train = []
	y_train = []
	for i in range(0, n_data):
		for j in range(0, 24-10):
			x_train.append(data[i*n_features:(i+1)*n_features, j:j+9]) # get previous 9 hours
			y_train.append(data[i*n_features+9][j+9]) # predict the 10-th hour
	return np.array(x_train), np.array(y_train)

	# return data

def train(batch_size, epochs, lr):
	global x_train
	global y_train
	global w
	global b

	iterations = x_train.shape[0] // batch_size

	prev_grad_w = 0.000001
	prev_grad_b = 0.000001

	prev_loss = -100

	for i in range(epochs):
		loss = 0
		for j in range(iterations):
			x = x_train[j*batch_size:(j+1)*batch_size]
			y = y_train[j*batch_size:(j+1)*batch_size]

			y_h = (x * w).sum(axis=1).sum(axis=1) + b
			delta = y-y_h
			grad_w = -(2 * delta[:,np.newaxis,np.newaxis] * x).sum(axis=0)
			grad_b = -(2 * delta).sum()
			
			# adadelta
			# prev_grad_w = lamb*prev_grad_w + (1-lamb)*(grad_w ** 2)
			# prev_grad_b = lamb*prev_grad_b + (1-lamb)*(grad_b ** 2)

			# adagrad
			prev_grad_w += (grad_w ** 2)
			prev_grad_b += (grad_b ** 2)

			w -= lr * grad_w / (prev_grad_w ** 0.5)
			b -= lr * grad_b / (prev_grad_b ** 0.5)

			loss += (delta ** 2).mean()
		loss /= iterations
		print ("{} epoch, loss = {}".format(i+1,loss))
		if abs(loss - prev_loss) < 0.00001:
			print ("Training finished")
			return
		prev_loss = loss


if __name__ == "__main__":
	path = 'train.csv'
	# data = preprocess(path)
	# print (data.shape)
	x_train, y_train = preprocess(path) # x_train.shape = (3360, 17, 9) , y_train.shape = (3360,)
	w = np.random.rand(18,9)
	b = np.random.rand(1)

	train(batch_size, epochs, lr)
	np.save('w.npy',w)
	np.save('b.npy',b)

