# import libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import cv2
import os
import random
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# define class/categories
cat = ['with_mask','without_mask']
print("Output - ", cat)

data = []
for category in cat:
    path = os.path.join('dataset',category)
    label = cat.index(category)
    # resize and add labels
    for file in os.listdir(path):
        img_path = os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        data.append([img,label])
print("Output - ",data)
print("The length of the dataset is ",len(data))

random.shuffle(data)
print("Output - ",data)

x = []
y = []
for features,label in data:
    x.append(features)
    y.append(label)

print("The number of images in x are ",len(x))
print("And the corresponding labels in y are ",len(y))

x = np.array(x)
y = np.array(y)

print("shape of x - ",x.shape)
print("shape of y - ",y.shape)

x = x/255
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)

# load vgg16 model
vgg = VGG16()
vgg.summary()

# create new sequential model
model = Sequential()

for layer in vgg.layers[:-1]:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.summary()

model.add(Dense(1,activation='sigmoid'))

model.summary()

# compile the model
opt = Adam(lr = 1e-4, weight_decay = 1e-4/32)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

# model training
history = model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))

# Plotting accuracy curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model 
model.save("faceMaskDetect_.h5")