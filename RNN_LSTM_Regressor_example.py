import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM,TimeDistributed
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006

def get_batch():
    global BATCH_START,TIME_STEPS
    # xs shape(50batch,20steps)
    xs = np.arange(BATCH_START,BATCH_START+BATCH_SIZE*TIME_STEPS).reshape(BATCH_SIZE,TIME_STEPS)/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    return  [seq[:,:,np.newaxis],res[:,:,np.newaxis],xs]

model = Sequential()

# build a LSTM RNN
model.add(LSTM(
    units=CELL_SIZE,
    batch_input_shape=(BATCH_SIZE,TIME_STEPS,INPUT_SIZE),
    return_sequences=True,
    stateful=True,
))

# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(lr=LR)
model.compile(optimizer=adam,loss='mse',)

print('Training..........')
for step in range(501):
    X_batch,Y_batch,xs = get_batch()
    cost = model.train_on_batch(X_batch,Y_batch)
    pred = model.predict(X_batch,BATCH_SIZE)
    plt.plot(xs[0,:],Y_batch[0].flatten(),'r',xs[0,:])
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)