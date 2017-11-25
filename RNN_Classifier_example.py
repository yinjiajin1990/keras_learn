import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,SimpleRNN
from keras.optimizers import Adam


TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called'
(X_train,y_train),(X_test,y_test)=mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1,28,28)/255.
X_test = X_test.reshape(-1,28,28)/255.
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# build model
model = Sequential()

# rnn cell
model.add(SimpleRNN(
    units=CELL_SIZE,
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(lr=LR)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# training
for step in range(12001):
    X_batch = X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:,:]
    Y_batch = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE,:]
    cost = model.train_on_batch(X_batch,Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost,accuracy = model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
        print('test cost:',cost,'test accuracy:',accuracy)
