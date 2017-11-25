import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam

#download the mnist to the path '~/.keras/datasets/' if it is first time to be called
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,1,28,28)/255.
X_test = X_test.reshape(-1,1,28,28)/255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# initail model
model = Sequential()

# add convolution2D
model.add(Convolution2D(
    batch_input_shape=(None,1,28,28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
))
model.add(Activation('relu'))

# add Maxpooling2D
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))

# add convolution2D
model.add(Convolution2D(
    64,5,strides=1,padding='same',data_format='channels_first'
))
model.add(Activation('relu'))

# add maxpooling2D
model.add(MaxPooling2D(2,2,padding='same',data_format='channels_first'))

# add full connection
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# add full connection
model.add(Dense(10))
model.add(Activation('softmax'))

# define optimizer
adm = Adam(lr=1e-4)

# compile model
model.compile(optimizer=adm,loss='categorical_crossentropy',metrics=['accuracy'])

print('Training..........')
# train model
model.fit(X_train,y_train,batch_size=32,epochs=1,)

print('Testing.........')
# test model
loss,accuracy = model.evaluate(X_test,y_test)
print('\ntest loss:',loss)
print('\ntest accuracy:',accuracy)

# predict
