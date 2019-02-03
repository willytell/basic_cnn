from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import keras
import pandas as pd
import numpy as np


def readFromExcel(filename, sheet_name, verbose=True):
    df = pd.read_excel(filename, sheet_name=sheet_name)

    # copying the columns with features
    x = df[df.columns[5:]].values

    # copying the column with the nodule diagnosis
    y = df[df.columns[3]].values

    if verbose:
        print("x.shape: {}".format(x.shape))
        print("y.shape: {}".format(y.shape))

    return x, y

print("\nReading train set:")
x_train, y_train = readFromExcel('/home/willytell/Escritorio/output/pipeline2A/extractedFeatures_Train.xlsx', 'Sheet1')

print("\nReading test set:")
x_test, y_test = readFromExcel('/home/willytell/Escritorio/output/pipeline2A/extractedFeatures_Test.xlsx', 'Sheet1')

# Only to simulate more data ;)
# x_train = np.random.random((6000, 88))
# y_train = np.random.randint(2, size=(6000, 1))


batch_size = 5
num_classes = 2
epochs = 3

nb_of_samples = x_train.shape[0]   # input: number of samples
nb_of_features = x_train.shape[1]  # input number of dimensions


# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], 88)
x_test = x_test.reshape(x_test.shape[0], 88)




# One-hot encoding of y_train labels (only execute once!)
#y_train = keras.utils.to_categorical(y_train, num_classes)
y_train = y_train.reshape(y_train.shape[0], 1)

#y_test = keras.utils.to_categorical(y_test, num_classes)
y_test = y_test.reshape((y_test.shape[0], 1))


# convert the data to the right type
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(type(x_train))


# model = Sequential()
# model.add(Dense(32, input_shape=(1, 88)))
# model.add(Flatten())
# #model.add(Dense(10, activation='relu', input_dim=10))
# #model.add(Dense(num_classes, activation='softmax'))
# model.add(Dense(2, activation='softmax'))

model = Sequential()
model.add(Dense(88, activation='relu', input_dim=88))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(), #'rmsprop',
              metrics=['accuracy'])

# model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs, verbose=1,
#     callbacks=None,
#     validation_split=0.2,
#     validation_data=None,
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs)
#score = model.evaluate(x_test, y_test, batch_size=batch_size)

#output = model.get_layer('dense_3').output
from keras.models import Model
model2 = Model(inputs=model.input, outputs=model.get_layer('dense_3').output)

#print(model2.shape)
dense_3_features = model2.predict(x_train)

print("dense_3_features.shape: {}".format(dense_3_features.shape))

print("end.")