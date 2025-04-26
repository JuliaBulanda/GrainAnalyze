import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2


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


mode = input("Enter 'train' to train the model or 'process' to process images: ")

if mode == 'train':
    X_train, y_train = load_data('training/original', 'training/mask')
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

elif mode == 'process':
    model = load_model('unet_disc_segmentation.keras')

    input_unet_path = 'input_unet'
    output_path = 'output_contours'
    os.makedirs(output_path, exist_ok=True)

    for image_file in os.listdir(input_unet_path):
        img_path = os.path.join(input_unet_path, image_file)
        img_loaded = load_img(img_path, target_size=(128,128))
        img_array = img_to_array(img_loaded) / 255.0

        prediction = model.predict(np.expand_dims(img_array, axis=0))[0].squeeze()
        print(f"Prediction stats for {image_file}: min={prediction.min()}, max={prediction.max()}, mean={prediction.mean()}")
        binary_mask = (prediction > 0.5).astype(np.uint8) * 255

        mask_debug_path = os.path.join(output_path, f"mask_{image_file}")
        cv2.imwrite(mask_debug_path, binary_mask)

        fullres_img = cv2.imread(img_path)
        fullres_img_rgb = cv2.cvtColor(fullres_img, cv2.COLOR_BGR2RGB)
        h, w, _ = fullres_img.shape

        mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"{image_file}: {len(contours)} contours found")

        contour_img = fullres_img_rgb.copy()
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            print(f"  Largest contour area: {area}")
            if area > 50:
                cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.title('Original Image')
        plt.imshow(fullres_img_rgb)

        plt.subplot(1,2,2)
        plt.title('Contours')
        plt.imshow(contour_img)

        plt.savefig(os.path.join(output_path, f"contour_{image_file}"))
        plt.close()
else:
    print("Invalid mode entered.")