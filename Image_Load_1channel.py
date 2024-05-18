#library
import os
from PIL import Image
import numpy as np
import cv2
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
import os
from PIL import Image
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from random import randrange
from keras.preprocessing.image import ImageDataGenerator

#Parameters
BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 50

#Load Data
def load_data():
    train_data = np.empty((480, 1, 256, 256), dtype="uint8")
    train_labels = np.empty((480,), dtype="uint8")
    test_data = np.empty((20, 1, 256, 256), dtype="uint8")
    test_labels = np.empty((20,), dtype="uint8")

    imgs_1 = os.listdir("./images")
    class_cnt = len(imgs_1)
    cnt = 0
    for i in range(0, class_cnt):
        for j in range(48):
            img_1 = cv2.imread("./images/" + str(i) + "/" + str(i) + " (" + str(j) + ")" + ".jpg")
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
            arr_1 = np.array(img_1)
            train_data[cnt, :] = [arr_1[:]]
            train_labels[cnt] = i 
            cnt += 1

    img_2 = os.listdir("./test_images")
    class_cnt1 = len(img_2)
    cnt1 = 0
    for i in range(0, class_cnt1):
        for j in range(2):
            img_2 = cv2.imread("./test_images/" + str(i) + "/" + str(i) + " (" + str(j) + ")" + ".jpg")
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
            arr_2 = np.array(img_2)
            test_data[cnt1, :] = [arr_1[:]]
            test_labels[cnt1] = i 
            cnt1 += 1
    return (train_data, train_labels, test_data, test_labels)
(train_data, train_labels, test_data, test_labels) = load_data()

#data preprocess
train_data = train_data / 255.0
train_labels = train_labels.flatten()
train_labels = to_categorical(train_labels, NUM_CLASSES)
test_data = test_data / 255.0
test_labels = test_labels.flatten()
test_labels = to_categorical(test_labels, NUM_CLASSES)

#model construct
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=(1, 256, 256), activation='relu', padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

#Model Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(),
    metrics=['acc'])


history = model.fit(
    train_data,
    train_labels,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    shuffle=True
)

#Plot Analysis
def show_train_history(train_type,test_type):
    plt.plot(history.history[train_type])
    plt.plot(history.history[test_type])
    plt.title('Train History')
    if train_type == 'acc':
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

score = model.evaluate(test_data, test_labels, verbose=1)
print('Train accuracy', history.history['acc'])
print('Train loss', history.history['loss'])
print('Test accuracy:', score[1])
print('Test loss:', score[0])

#Save Model
try:
    if score[0] > 0.8:
        model.save_weights("finger{score[1]}.h5")
        print("success save model weights")
except:
    print("error save model weights")

