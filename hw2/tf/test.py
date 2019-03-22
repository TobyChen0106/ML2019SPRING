import numpy as np
import math
import csv
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization

def preprocess(A):
    augmented_weight = 1
    new_x = np.empty((A.shape[0], A.shape[1] + (augmented_weight - 1) * 5))
    
    A[:,[2,5]] = A[:, [5,2]]
    
    for i in range(5):
        for j in range(augmented_weight):
            new_x[:, i*augmented_weight + j] = np.power(A[:, i], j + 1)
    new_x[:, 5*augmented_weight:] = A[:, 5:]
    # new_x[:, augmented_weight*5] = A[:,2]
    # print (new_x[0])            

    return new_x

if __name__ == "__main__":

    x_test = np.genfromtxt('X_test', delimiter=',')[1:]
    x_test = preprocess(x_test)
    x_test = preprocess(x_test)
    
    model = Sequential()
    model.add(Dense(1000, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr=5e-4)
    model.checkpoint = ModelCheckpoint('best_29.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=500, batch_size=50, shuffle = True, validation_split = 0.2)

    scores = model.evaluate(x_train, y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    
    # try:
    #     gradient_decent(x_train, y_train)
    # except (KeyboardInterrupt):
    #     np.save('result/w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
    #     np.save('result/b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
    #     np.save('w',w)
    #     np.save('b',b)
    #     np.save('w_v',w_var)
    #     np.save('b_v',b_var)

    # np.save('w_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), w)
    # np.save('b_'+str(lr)+'_'+str(epoch)+'_'+str(batch)+'_l='+str(loss), b)
    # np.save('w',w)
    # np.save('b',b)
