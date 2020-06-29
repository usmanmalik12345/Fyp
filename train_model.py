import numpy as np
from alexnet import alexnet
from tensorflow.keras.callbacks import TensorBoard
from random import shuffle
import cv2
import pandas


WIDTH = 80
HEIGHT = 60


MODEL_NAME = 'dummy.model'
#tensorboard = TensorBoard(log_dir = 'logs/{}'.format(MODEL_NAME))
train_data = np.load('Combined_dataset_v4(6_7_8_9).npy', allow_pickle=True)
shuffle(train_data)

train = train_data[:-1000]
test = train_data[-1000:]

print(len(np.array([i[0] for i in train])))
print(len(np.array([i[0] for i in train]).reshape(-1,80,60,1)))
#print(len(np.array([i[1] for i in train])))

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

shape = X.shape[1:]
model = alexnet(shape)

X = np.array(X)
Y = np.array(Y)

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

test_x = np.array(test_x)
test_y = np.array(test_y)
#model.fit(X, Y, batch_size=64, epochs=1 ,validation_data = (test_x,test_y))#,callbacks = [tensorboard])
#model.save('delete_this.model')
print(Y[:10])













