import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions

from google.colab import drive
drive.mount('/content/drive')

path='drive/tomotodataset/archive(6)/Tomato Leaf Diseases'

len(os.listdir("/content/drive/MyDrive/tomotodataset/archive (6)/Tomato Leaf Diseases/Training Set"))


train_datagen=ImageDataGenerator(zoom_range=0.5,shear_range=0.3,horizontal_flip=True,preprocessing_function=preprocess_input)
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train=train_datagen.flow_from_directory(directory="/content/drive/MyDrive/tomotodataset/archive (6)/Tomato Leaf Diseases/Training Set",
                                        target_size=(256,256),
                                        batch_size=32)
val=val_datagen.flow_from_directory(directory="/content/drive/MyDrive/tomotodataset/archive (6)/Tomato Leaf Diseases/Validation Set",
                                        target_size=(256,256),
                                        batch_size=32)

t_img,lable=train.next()

def plotImage(img_arr,lable):
  for im ,l in zip(img_arr ,lable):
    plt.figure(figsize=(5,5))
    #plt.imshow(im)
    plt.show()
    
    plotImage(t_img[:5],lable[:3])
    
   // Bubilling model
  
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras


base_model=VGG19(input_shape=(256,256,3),include_top=False)

for layer in base_model.layers:
  layer.trainable=False
  
  base_model.summary()
  
  x=Flatten()(base_model.output)
x=Dense(units=38,activation='softmax')(x)

//creating our model

model=Model(base_model.input,x)

model.summary()
model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

//Early stopping and model check point

from keras.callbacks import ModelCheckpoint,EarlyStopping
es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=3,verbose=1)
mc=ModelCheckpoint(filepath="best_model.h5",
                    monitor="val_accuracy",
                    min_delta=0.1,
                    verbose=1,
                    save_best_only=True)
cb=[es,mc]


his=model.fit_generator(val,
                        steps_per_epoch=16,
                        epochs=50,
                        verbose=1,
                        callbacks=cb,
                        validation_data=val,
                        validation_steps=16)


h=history
h.keys()
