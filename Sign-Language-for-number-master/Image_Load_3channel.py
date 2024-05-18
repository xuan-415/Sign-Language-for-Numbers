#library
from library import *

#Parameters
BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 25

#Load Data
def load_data():
    train_data = np.empty((2420, 3, 256, 256), dtype="uint8")
    train_labels = np.empty((2420,), dtype="uint8")
    val_data = np.empty((610, 3, 256, 256), dtype="uint8")
    val_labels = np.empty((610,), dtype="uint8")
    test_data = np.empty((240, 3, 256, 256), dtype="uint8")
    test_labels = np.empty((240,), dtype="uint8")

    imgs_1 = os.listdir("./images")
    class_cnt = len(imgs_1)
    cnt = 0
    cnt2 = 0
    for i in range(0, class_cnt):
        for j in range(303):
            if j > 60:
                img_1 = Image.open("./images/" + str(i) + "/" + str(i) + " (" + str(j) + ")" + ".jpg")
                arr_1 = np.array(img_1)
                train_data[cnt, :, :, :] = [arr_1[:, :, 0], arr_1[:, :, 1], arr_1[:, :, 2]]
                train_labels[cnt] = i 
                cnt += 1
            else:
                img_3 = Image.open("./images/" + str(i) + "/" + str(i) + " (" + str(j) + ")" + ".jpg")
                arr_3 = np.array(img_3)
                val_data[cnt2, :, :, :] = [arr_3[:, :, 0], arr_3[:, :, 1], arr_3[:, :, 2]]
                val_labels[cnt2] = i 
                cnt2 += 1 

    img_2 = os.listdir("./test_images")
    class_cnt1 = len(img_2)
    cnt1 = 0
    for i in range(0, class_cnt1):
        for j in range(24):
            img_2 = Image.open("./test_images/" + str(i) + "/" + str(i) + " (" + str(j) + ")" + ".jpg")
            arr_2 = np.array(img_2)
            test_data[cnt1, :, :, :] = [arr_2[:, :, 1], arr_2[:, :, 2], arr_2[:, :, 0]]
            test_labels[cnt1] = i 
            cnt1 += 1

    return (train_data, train_labels, test_data, test_labels, val_data, val_labels)
(train_data, train_labels, test_data, test_labels, val_data, val_labels) = load_data()

#data preprocessing
def data_preprocessing(data, labels):
    data   = data.reshape(data.shape[0], data.shape[1], data.shape[2], 256)
    data   = data / 255.0
    labels = labels.flatten()
    labels = to_categorical(labels, NUM_CLASSES)
    return (data, labels)

(train_data, train_labels)   = data_preprocessing(train_data, train_labels) 
(val_data, val_labels)       = data_preprocessing(val_data, val_labels) 
(test_data, test_labels)     = data_preprocessing(test_data, test_labels) 

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

model = create_model()
plot_model(model, to_file="model.png", show_shapes=True)

#Data augmention
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False
)
datagen.fit(train_data)

#Model Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(),
    metrics=['acc'])

history = model.fit(datagen.flow(train_data, train_labels, batch_size=BATCH_SIZE), 
                    validation_data=(val_data, val_labels),
                    epochs=EPOCHS,
                    verbose=1,
                    steps_per_epoch= train_data.shape[0] / BATCH_SIZE,
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
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

score = model.evaluate(test_data, test_labels, verbose=1)
y_true = model.predict(test_data)
y_true = np.round(y_true)
print(classification_report(test_labels, y_true))
print('Train accuracy', history.history['acc'])
print('Train loss', history.history['loss'])
print('Test accuracy:', score[1])
print('Test loss:', score[0])

#Save Model
try:
    model.save_weights(f"finger{score[1]}.h5")
    print("success save model weights")
except:
    print("error save model weights")


