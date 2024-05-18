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
from sklearn.metrics import classification_report
NUM_CLASSES = 10