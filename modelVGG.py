import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import layers, models
from keras.preprocessing import image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224,1))

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=10,  
    steps_per_epoch=len(train_generator),
)
# Prepare your validation dataset similarly to the training dataset
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    'data/Validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


validation_loss, validation_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {validation_accuracy*100:.2f}%')

model.save('xray_classification_model.h5')

