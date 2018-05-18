# Import model
from keras.models import Sequential
# Import layers
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical  # to one-hot
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np


def loadData(path):
    filepath = './data/'
    tr_data = np.load(filepath + path)
    x = np.asarray(tr_data["mfcc"], dtype=np.float32)
    y = np.asarray(tr_data["genre"], dtype=np.int32)
    # one hot encode
    y = to_categorical(y)
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    return x, y


class LossAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.files = 0
        self.size = 0
        self.losses = []
        self.lossesVal = []
        self.accs = []
        self.accsVal = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lossesVal.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.accsVal.append(logs.get('val_acc'))
        self.size = self.size + 1

        if self.size > 5000:
            np.save('./loss'+str(self.files), self.losses)
            np.save('./lossVal'+str(self.files), self.lossesVal)
            np.save('./acc'+str(self.files), self.accs)
            np.save('./accVal'+str(self.files), self.accsVal)
            self.losses = []
            self.lossesVal = []
            self.accs = []
            self.accsVal = []
            self.size = 0
            self.files = self.files + 1



num_epochs = 20000
num_classes = 5
batch_size = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 13),
                 activation='relu',
                 padding = 'valid',
                 input_shape=(300, 13, 1) ))
model.add(MaxPooling2D(pool_size=(3, 1), strides=(3,1), padding = 'valid'))
model.add(Conv2D(32, kernel_size=(3, 32), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 1), strides=(3,1)))
model.add(Conv2D(32, kernel_size=(3, 32), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(33, 1), strides=(33,1)))
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(units=num_classes, activation='softmax'))

# print(model.summary()) # prints sizes and structure of network

# Learning process
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  # optimizers = ['sgd', 'adam']
              metrics=['accuracy'])

# Normal features
x_train, y_train = loadData('dataTrainMFCC_5C.npz')
print(y_train.shape, np.max(y_train), np.min(y_train))
print(x_train.shape, np.max(x_train), np.min(x_train))
x_train = np.reshape(x_train, [350, 300, 13, 1])
x_val, y_val = loadData('dataValMFCC_5C.npz')
print(y_val.shape, np.max(y_val), np.min(y_val))
print(x_val.shape, np.max(x_val), np.min(x_val))
x_val = np.reshape(x_val, [50, 300, 13, 1])
x_test, y_test = loadData('dataTestMFCC_5C.npz')
print(y_test.shape, np.max(y_test), np.min(y_test))
print(x_test.shape, np.max(x_test), np.min(x_test))
x_test = np.reshape(x_test, [100, 300, 13, 1])

# Callback hook to store data per epoch
history = LossAccHistory()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=num_epochs, batch_size=batch_size, callbacks=[history])

# Evaluate your performance
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# print(loss_and_metrics)
# # Generate predictions
# classes = model.predict(x_test, batch_size=128)
# predics = np.argmax(classes, axis=1)

# save the model
model.save('./keras_model.h5')
# del model

# Save values
np.save('./loss', history.losses)
np.save('./lossVal', history.lossesVal)
np.save('./acc', history.accs)
np.save('./accVal', history.accsVal)
