import keras.layers
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from keras.layers import Input, Lambda, Dense, Flatten, Layer
from keras.models import Model
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D,
                                     BatchNormalization, Permute, TimeDistributed, GlobalAveragePooling2D,
                                     SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D, Conv2DTranspose,
                                     ReLU, UpSampling2D, Concatenate, Conv2DTranspose)

# create dataFrame
ROOT_PATH = './data/data6/Garbage classification/Garbage classification/'
dataPath = {}
for i in os.listdir(ROOT_PATH):
    for j in os.walk(ROOT_PATH + i):
        for k in j[2]:
            dataPath[ROOT_PATH + i + '/' + k] = i

dataPath = pd.DataFrame(dataPath.items(), columns=['path', 'class_'])
dataPath = dataPath.sample(frac=1)


# split Data
def split_data(data, ratio):
    last = int(len(data) * ratio)
    return data[:last], data[last:]


train, test = split_data(dataPath, .8)

train.to_csv('./data/data6/train.csv', index=False)
test.to_csv('./data/data6/test.csv', index=False)

# check the balancing
print(train["class_"].value_counts())

# dividing between test and validation
train, validation = split_data(train, 0.9)
print(train["class_"].value_counts)
print(validation["class_"].value_counts)
print(test["class_"].value_counts)

# generation of autoencoder
# number images per lot
batch_size = 16
# image Size
size = 224
# number of time the NN will check all the data : normal one : 50
epoch = 2
# ImageDataGenerator generate, improve, resize, normalize and shake images before cnn
trainDataGenerator = ImageDataGenerator(rescale=1./255)
trainGenerator = trainDataGenerator.flow_from_dataframe(
    dataframe=train,
    x_col='path',
    y_col='class_',
    target_size=(size, size),
    batch_size=batch_size,
    class_mode="input"
    )
validationDataGenerator = ImageDataGenerator(rescale=1./255)
validationGenerator = validationDataGenerator.flow_from_dataframe(
    dataframe=validation,
    x_col='path',
    y_col='class_',
    target_size=(size, size),
    batch_size=batch_size,
    class_mode="input"
    )


# Squeeze and Excitation function
def se_block_enc(inputs, alpha):
    input_channels = inputs.shape[-1]
    tensor = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    tensor = tf.keras.layers.Dense(units=alpha, activation="relu")(tensor)
    tensor = tf.keras.layers.Dense(units=input_channels, activation="sigmoid")(tensor)
    # tensor = tf.reshape(tensor, [-1, 1, 1, input_channels])
    tensor = LayerKerasTensorToTfTensor()(tensor=tensor, input_channels=input_channels)
    tensor = inputs * tensor
    return tensor


class LayerKerasTensorToTfTensor(Layer):
    def call(self, tensor, input_channels):
        return tf.reshape(tensor, [-1, 1, 1, input_channels])


# Auto Encoder : first the encoder to change the data, and then the decoder
# ENCODER
inputImage = Input(shape=(size, size, 3))  # image size, and 3 because color image
x = Conv2D(48, (3, 3), activation='relu', padding='same')(inputImage)
x = se_block_enc(x, 20)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = se_block_enc(x, 30)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = se_block_enc(x, 50)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(32, (1, 1), activation='relu', padding='same')(x)

# Bottleneck
latentSize = (28, 28, 32)

# DECODER
direct_input = Input(shape=latentSize)
x = Conv2D(192, (1, 1), activation='relu', padding='same')(direct_input)
x = UpSampling2D((2, 2))(x)  # make the tensor bigger
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# COMPILE
encoder = Model(inputImage, encoded)
decoder = Model(direct_input, decoded)
autoencoder = Model(inputImage, decoder(encoded))


# Try to compile
autoencoder.compile(optimizer=tf.keras.optimizers.Adamax(), loss='binary_crossentropy')
history = autoencoder.fit(trainGenerator, validation_data=validationGenerator, epochs=epoch, verbose=2)

#loss function
epochs = list(range(len(history.history['loss'])))
fig, ax = plt.subplots(1, 2)

train_loss = history.history['loss']

val_loss = history.history['val_loss']

fig.set_size_inches(20, 10)

ax[1].plot(epochs, train_loss, 'g-o', color='r', label='Training Loss')
ax[1].plot(epochs, val_loss, 'go-', label='Validation Loss')
ax[1].set_title('Training Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training Loss")
plt.show()

autoencoder.save("./model/auto_encoder.keras")
encoder.save('./model/encoder.keras')
decoder.save('./model/decoder.keras')

# Check performance auto Encoder with an image
orig = cv2.imread(test.iloc[0]['path'])
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
orig = orig * 1.0 / 255.0
orig = cv2.resize(orig, (size, size), interpolation=cv2.INTER_NEAREST)
img = tf.expand_dims(orig, axis=0)

encoder_output = encoder.predict(img)
plt.title('Original')
plt.imshow(orig)
plt.show()

print(encoder_output.shape)

# Display the encoded image
plt.imshow(encoder_output[0][0])  # a channel output shown in plot
plt.title("a sample output of encoded image")
plt.show()

# and decode
original_output = decoder.predict(encoder_output)
plt.imshow(original_output[0])
plt.title("reconstructed image")
plt.show()

# freezing the encoder
for i in encoder.layers:
    i.freeze = True


# image generator with data augmentation
trainDataGenerator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
trainImages = trainDataGenerator.flow_from_dataframe(
    dataframe=train,
    x_col='path',
    y_col='class_',
    batch_size=batch_size,
    target_size=(size, size),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=42,
)
validImages = trainDataGenerator.flow_from_dataframe(
    dataframe=validation,
    x_col='path',
    y_col='class_',
    batch_size=batch_size,
    target_size=(size, size),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=42,
)


import tensorflow_hub as hub
def get_from_hub(model_url):
    inputs = tf.keras.Input((224, 224, 3))
    hub_module = hub.KerasLayer(model_url,trainable=False)
    outputs = hub_module(inputs)
    return tf.keras.Model(inputs, outputs)