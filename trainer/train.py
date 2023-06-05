# import modules and libraries
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
import splitfolders 
import matplotlib.pyplot as plt

# split train validation
splitfolders.ratio("/gcs/bucket-training-model/dataset\ bangkit", output="food-data", seed=1337, ratio=(.8, .2), group_prefix=None) 

training_dir = os.path.join('food-data/', 'train')
testing_dir = os.path.join('food-data/', 'val')

training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	training_dir,
	target_size=(224, 224),
  color_mode='rgb',
	class_mode='categorical',
  batch_size=32,
  shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
	testing_dir,
	target_size=(224, 224),
  color_mode='rgb',
	class_mode='categorical',
  batch_size=32,
  shuffle=True,
)

# build model
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False

inputs = pretrained_model.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(21, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

print(model.summary())

#callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.96):
      print("\nReached 96% accuracy so cancelling training!")
      self.model.stop_training = True

# compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[myCallback()]
)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

# save model
model.save("/gcs/bucket-training-model/foodModel1.h5")
print('Model Saved!')