from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D, Dropout
from keras.models import Sequential
import matplotlib.pylab as plt

batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_x = 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:, 5, :]
x_test = x_test[:, 5, :]

print(x_train.shape)
print(x_test.shape)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 1)

input_shape = (x_train.shape[0], img_x, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below

# One-hot encoding of y_train labels (only execute once!)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv1D(1, 1, activation='relu', input_shape=(28, 1)))
#model.add(Conv1D(64, 3, activation='relu'))
#model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=3)
#score = model.evaluate(x_test, y_test, batch_size=16)