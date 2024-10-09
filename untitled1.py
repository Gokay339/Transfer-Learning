

from keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import cifar10      # dataset
import cv2
import numpy as np


# %%

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # dataset çağırma
print("x_train.shape", x_train.shape)

numberOfClass = 10
y_train = to_categorical(y_train, numberOfClass)
y_test = to_categorical(y_test, numberOfClass)


input_shape = x_train.shape[1:]

# %%

plt.imshow(x_train[2].astype(np.uint8))
plt.axis("off")
plt.show()


#%%

def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage,48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48,3))  #opencv metodunu çağırarak boyutlarını 48x48 yaptık
    return new_array

x_train = resize_img(x_train)
print("x_train shape : ",x_train.shape)
x_test = resize_img(x_test)
y_train = resize_img(y_train)
y_test = resize_img(y_test)


plt.figure()
plt.imshow(x_train[2].astype(np.uint8))
plt.axis("off")
plt.show()

#%%

vgg = VGG19(include_top=False,weights="imagenet",input_shape=(48,48,3))
#weights imagenet anlamı benim weightlerim imagenet ile eğitilmiş olsun
#include_top False anlamı output katmanını çıkardığı anlamına gelir

print(vgg.summary()) #parametreleri verir

vgg_layer_list = vgg.layers
print(vgg_layer_list)


model = Sequential()

for layer in vgg_layer_list:
    model.add(layer)
    
for layer in model.layers:
    layer.trainable = False # Modelleri Train etme
    
model.add(Flatten())  # 2d veya 3d durumuna göre düzleme yapar
model.add(Dense(128)) # 128 nöron
model.add(Dense(numberOfClass,activation = "softmax"))  # output


model.compile(loss="categorical_crossentropy",
              optimizer = "rmsprop",
              metrics =["accuracy"])


hist = model.fit(x_train,y_train,validation_split=0.2,epochs = 5,batch_size =1000)
# validation_split=0.2 demek, eğitim verisinin %20'sini doğrulama seti olarak ayırır, 
# geri kalan %80'ini ise eğitim için kullanır.


#%%

model.save_weights("deneme.h5")  # weightleri kaydeder

print(hist.history.keys())

plt.plot(hist.history["loss"], label="Training Loss")
plt.plot(hist.history["val_loss"],label=" Validation Loss")
plt.legend()
plt.show()


plt.plot(hist.history["accuracy"],label="Training Accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation Accuracy")
plt.legend()
plt.show()

#%%

import json , codecs

with open("deneme.json","w") as f:
    json.dump(hist.history,f)







