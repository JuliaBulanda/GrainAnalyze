import os
import numpy as np
import matplotlib.pyplot as plt

# Wyłącz optymalizacje oneDNN dla TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"    #musi być przed importem

import tensorflow as tf

csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=False)

orgilan_path='training/original'
size=(512, 512, 3)    # I don't know how it works, but can be helpful
    # smoler size = faster building model, but worse shape of result
    #too big value use a lot of RAM !!!
    #3. number is number of colors

# Funkcja do ładowania i przetwarzania danych
def load_data(image_path, mask_path, target_size=(size[0], size[1])):
    images = []
    masks = []

    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))

    for img, msk in zip(image_files, mask_files):
        img_loaded = tf.keras.preprocessing.image.load_img(os.path.join(image_path, img), target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img_loaded) / 255.0

        mask_loaded = tf.keras.preprocessing.image.load_img(os.path.join(mask_path, msk), target_size=target_size, color_mode='grayscale')
        mask_array = tf.keras.preprocessing.image.img_to_array(mask_loaded) / 255.0
                #zastanawiam się czy do tego miejsca nie jest lepsze cv2 ???
        mask_array = (mask_array > 0.5).astype(np.float32)

        images.append(img_array)
        masks.append(mask_array)

    return np.array(images), np.array(masks)

# Funkcja definiująca model U-Net
def unet(input_size=size):
    inputs = tf.keras.layers.Input(input_size)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)

    # Decoder
    up2 = tf.keras.layers.Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same')(conv3)
    concat2 = tf.keras.layers.concatenate([conv2, up2])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)

    up1 = tf.keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(conv4)
    concat1 = tf.keras.layers.concatenate([conv1, up1])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

def train():
    # Ładowanie danych treningowych
    X_train, y_train = load_data('training/original', 'training/mask')
    print(f"Loaded {len(X_train)} images for training.")
    plt.imshow(y_train[0].squeeze(), cmap='gray')
    plt.title("First training mask")
    plt.show()

    # Kompilacja modelu
    model = unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callback do zapisywania najlepszego modelu
    checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_disc_segmentation.keras',
                                                    monitor='val_loss',
                                                    save_best_only=True)

    # Trenowanie modelu
    model.fit(X_train, y_train,
              batch_size=1,
              steps_per_epoch=max(1, len(X_train)),
              epochs=50,
              validation_split=0.2,
              shuffle=True,
              callbacks=[checkpoint, csv_logger])



if __name__=="__main__":
    train()