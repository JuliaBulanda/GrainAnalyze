import os
import numpy as np

# Wyłącz optymalizacje oneDNN dla TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    #musi być przed importem

import tensorflow as tf

import cv2



# Wczytanie modelu
model = tf.keras.models.load_model('unet_disc_segmentation.keras')

# Ścieżki wejścia i wyjścia
input_unet_path = 'input_unet'
output_path = 'output_contours'
target_size=(512, 512)

def process(input_unet_path = 'input_unet', output_path = 'output_contours'):
    os.makedirs(output_path, exist_ok=True)

        # Przetwarzanie obrazów wejściowych
        # Przetwarzanie obrazów wejściowych
        # for dirpath, dirnames, filenames in os.walk(input_unet_path):
        #     for image_file in filenames:
                # Ścieżka do obrazu
                img_path = os.path.join(dirpath, image_file)
                img_loaded = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img_loaded) / 255.0

                # Predykcja maski
                prediction = model.predict(np.expand_dims(img_array, axis=0))[0].squeeze()
                print(f"Prediction stats for {image_file}: min={prediction.min()}, max={prediction.max()}, mean={prediction.mean()}")
                binary_mask = (prediction > 0.5).astype(np.uint8) * 255

                # # Zapis maski binarnej do pliku
                # mask_debug_path = os.path.join(output_path, f"mask_{image_file}")
                # cv2.imwrite(mask_debug_path, binary_mask)

                # Przygotowanie obrazu w pełnej rozdzielczości
                fullres_img = cv2.imread(img_path)
                fullres_img_rgb = cv2.cvtColor(fullres_img, cv2.COLOR_BGR2RGB)
                h, w, _ = fullres_img.shape

                mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)



                # Znalezienie konturów
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"{image_file}: {len(contours)} contours found")

                contour_img = fullres_img_rgb.copy()
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    print(f"  Largest contour area: {area}")
                    if area > 50:
                        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 2)

                pred_min = float(prediction.min())
                pred_max = float(prediction.max())
                pred_mean = float(prediction.mean())

                num_contours = len(contours)
                largest_area = 0.0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_area = float(cv2.contourArea(largest_contour))

                csv_writer.writerow([
                    image_file, pred_min, pred_max, pred_mean, num_contours, largest_area
                ])


                cv2.imwrite(os.path.join(output_path, f"contour_{image_file}"), contour_img)


if __name__=="__main__":
    process(input_unet_path, output_path)