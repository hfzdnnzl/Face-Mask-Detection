# import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
start = time.time()

# initialize
INIT_LR = 1e-4
EPOCHS = 20
BATCHSIZE = 32

DIRECTORY = r'C:\Users\60166\OneDrive\Documents\UM - No Lyfe\Semester 6\WIX3001\image classifier\Train Image Classifier'
CATEGORIES = ['correctly masked','incorrectly masked']

# grab the image, transform and label it
print('[INFO] loading images...')

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        # read image
        img_path = os.path.join(path, img)
        image = load_img(img_path,target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        # append image data and labels
        data.append(image)
        labels.append(category)

# pre-processing
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype='float32')
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data,labels,
                                                test_size=0.2,
                                                stratify=labels,
                                                random_state=1)

# construct image generator
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# base model mobileNetV2
baseModel = MobileNetV2(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(224,224,3)))

# head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128,activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation='softmax')(headModel)

# construct the model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze the base model
for layer in baseModel.layers:
    layer.trainable = False

# compile model
print('[INFO] compiling model...')
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print('[INFO] training head...')
H = model.fit(aug.flow(trainX,trainY,batch_size=BATCHSIZE),
              steps_per_epoch = len(trainX)//BATCHSIZE,
              validation_data=(testX,testY),
              validation_steps=len(testX)//BATCHSIZE,
              epochs=EPOCHS)

# make predictions
print('[INFO] evaluating network...')
predIdxs = model.predict(testX,batch_size=BATCHSIZE)
predIdxs = np.argmax(predIdxs, axis=1)

# show report
print(classification_report(testY.argmax(axis=1),predIdxs,target_names=lb.classes_))

# save model
print('[INFO] saving detector model...')
model.save('mask_detector.model', save_format='h5')

# plot the training loss and accuracy
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,N), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0,N), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0,N), H.history['accuracy'], label = 'train_accuracy')
plt.plot(np.arange(0,N), H.history['val_accuracy'], label = 'val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('plt.png')

print("--- %s minutes ---" % ((time.time() - start)/60))