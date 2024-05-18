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
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros
from sklearn import preprocessing

#model construct
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(3, 256, 256), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return model

#model construct
model = create_model()
model.load_weights("finger1.85.h5")

def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image,cmap='binary')
    plt.show()

img=np.array(Image.open('test.jpg').convert('L'))
plot_image(img)

# 樣本前處理
x_Test = img.reshape(65536).astype('float32')
x_Test_normalize = x_Test.astype('float32') / 255.0

# 樣本預測
prediction=model.predict(x_Test_normalize)
print(prediction[0])
print(np.argmax(prediction, axis=1))
