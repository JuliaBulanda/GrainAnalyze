import os

# Wyłącz optymalizacje oneDNN dla TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    #musi być przed importem


import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# Load and preprocess data
def load_data(image_path, mask_path, target_size=(128,128)):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))

    for img, msk in zip(image_files, mask_files):
        img_loaded = load_img(os.path.join(image_path, img), target_size=target_size)
        img_array = img_to_array(img_loaded) / 255.0

        mask_loaded = load_img(os.path.join(mask_path, msk), target_size=target_size, color_mode='grayscale')
        mask_array = img_to_array(mask_loaded) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    return np.array(images), np.array(masks)


# Define U-Net model
def unet(input_size=(128,128,3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    up2 = Conv2DTranspose(128, 3, strides=(2,2), activation='relu', padding='same')(conv3)
    concat2 = concatenate([conv2, up2])

    conv4 = Conv2D(128, 3, activation='relu', padding='same')(concat2)

    up1 = Conv2DTranspose(64, 3, strides=(2,2), activation='relu', padding='same')(conv4)
    concat1 = concatenate([conv1, up1])

    conv5 = Conv2D(64, 3, activation='relu', padding='same')(concat1)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model



X_train, y_train = load_data('training/original', '../training/mask')
print(f"Loaded {len(X_train)} images for training.")
plt.imshow(y_train[0].squeeze(), cmap='gray')
plt.title("First training mask")
plt.show()

model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('unet_disc_segmentation.keras', monitor='val_loss', save_best_only=True)

model.fit(X_train, y_train,
          batch_size=1,
          steps_per_epoch=max(1, len(X_train)),
          epochs=50,
          validation_split=0.2,
          shuffle=True,
          callbacks=[checkpoint])