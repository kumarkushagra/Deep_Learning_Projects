import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv2D, BatchNormalization, ReLU, Add, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from PIL import Image
import os

# Custom generator to handle image loading errors
class SafeDataGenerator(Sequence):
    def __init__(self, generator):
        self.generator = generator
        self.batch_size = generator.batch_size

    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        batch_x, batch_y = self.generator[index]
        safe_batch_x = []
        safe_batch_y = []

        for img, label in zip(batch_x, batch_y):
            try:
                Image.fromarray((img * 255).astype('uint8'))  # Verify the image
                safe_batch_x.append(img)
                safe_batch_y.append(label)
            except Exception as e:
                print(f"Skipping image due to error: {e}")

        if len(safe_batch_x) == 0:
            raise ValueError("All images in the batch are invalid.")

        return np.array(safe_batch_x), np.array(safe_batch_y)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r"D:\Development\Dataset\Bone_Fracture\train",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    r"D:\Development\Dataset\Bone_Fracture\train",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_generator = datagen.flow_from_directory(
    r"D:\Development\Dataset\Bone_Fracture\test",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

train_generator_safe = SafeDataGenerator(train_generator)
val_generator_safe = SafeDataGenerator(val_generator)

def resnet(x, filters, kernel_size=3, stride=1, use_shortcut=True):
    shortcut = x
    # First convolution layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    if use_shortcut:
        # Adjust shortcut if the number of filters has changed
        shortcut = Conv2D(filters, kernel_size=1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut to the output of the first convolution layer
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x

inputs = Input(shape=(256, 256, 3))

# Starting the model creation
x = Conv2D(32, (3, 3))(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

x = resnet(x, 32)

x = Flatten()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint(
    r'D:\PROJECT\Deep_Learning_Projects\Fracture_Segmentation\model_checkpoint.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

history = model.fit(
    train_generator_safe,
    steps_per_epoch=len(train_generator_safe),
    validation_data=val_generator_safe,
    validation_steps=len(val_generator_safe),
    epochs=20,
    callbacks=[checkpoint_callback]
)
