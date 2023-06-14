# import modules and libraries
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import splitfolders 
from keras.preprocessing.image import ImageDataGenerator
from keras import Model 
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from google.cloud import storage

credentials_path = 'credentials.json'
bucket_name = "bucket-training-model"

# split train validation
splitfolders.ratio(f"/gcs/{bucket_name}/dataset bangkit", output= f"/gcs/{bucket_name}/food-data", seed=1337, ratio=(.8, .2), group_prefix=None) 

training_dir = os.path.join(f'/gcs/{bucket_name}/food-data/', 'train')
testing_dir = os.path.join(f'/gcs/{bucket_name}/food-data/', 'val')

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
InceptionV3_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# The last 15 layers fine tune
for layer in InceptionV3_model.layers[:-15]:
    layer.trainable = False

x = InceptionV3_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.3)(x)
output  = Dense(units=21, activation='softmax')(x)
model = Model(InceptionV3_model.input, output)

print(model.summary())

#callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.90):
      print("\nReached 90% accuracy so cancelling training!")
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

plt.figure()  # Membuat figure baru
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
# plt.savefig(f'/gcs/{bucket_name}/GrafikModel1.png')  # Simpan gambar sebelum memanggil plt.show()
# plt.show()

# Simpan gambar ke file lokal
local_file_path = 'grafik.png'
plt.savefig(local_file_path)

# Unggah gambar ke bucket GCS
client = storage.Client.from_service_account_json(credentials_path)
bucket = client.get_bucket(bucket_name)
blob = bucket.blob('GrafikModel1.png')
blob.upload_from_filename(local_file_path)

# Hapus file lokal setelah diunggah
os.remove(local_file_path)

# save model
model.save(f'/gcs/{bucket_name}/foodModel1.h5')
print('Model Saved!')