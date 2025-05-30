#aktualna wersja 28-05

import warnings
import numpy as np
import tensorflow as tf
import cv2

# Załaduj model UNet
model = tf.keras.models.load_model('unet_disc_segmentation.keras')
target_size = (512, 512)

def crop_disk_from_image(img_path):    #, save_mask_path="mask_output.png", show_mask=True):
    """
    Wczytuje obraz, generuje maskę UNet-em, znajduje największy kontur (dysk),
    a potem zwraca przycięty do kwadratu wycinek oryginalnego obrazu.
    """
    # 1. Wczytanie oryginału
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Nie udało się wczytać {img_path}")
    h, w = img.shape[:2]

    # 2. Przygotowanie tensora do UNet-a (zmiana rozmiaru i normalizacja)

    resized = cv2.resize(img, target_size)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # (1,H,W,3)

    # 3. Predykcja maski
    pred = model.predict(tensor)[0, ..., 0]
    mask = (pred > 0.5).astype(np.uint8) * 255


    # 4. Przywrócenie do pełnej rozdzielczości
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # cv2.imwrite(save_mask_path, mask_full)

    # 5. Detekcja konturów
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if not contours:
            warnings.warn(f"Nie znaleziono konturów na masce dla obrazu {img_path}. Zwracany jest cały obraz.")
            return img  # Zwracamy oryginalny obraz

    # 6. Wybór największego konturu i bounding box
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest_contour)

    # 7. Korekcja bounding box do kwadratu
    side = max(cw, ch)  # kwadrat ma bok równy największemu wymiarowi bounding box
    cx, cy = x + cw // 2, y + ch // 2  # środek bounding box

    x0 = max(0, cx - side // 2 - 20)
    y0 = max(0, cy - side // 2 - 20)
    x1 = min(w, x0 + side + 20)
    y1 = min(h, y0 + side + 20)

    # Jeśli kwadrat nie mieści się w obrazie, przesuwamy go
    if x1 > w:
        x0 -= (x1 - w)
        x1 = w
    if y1 > h:
        y0 -= (y1 - h)
        y1 = h
    if x0 < 0:
        x1 += abs(x0)
        x0 = 0
    if y0 < 0:
        y1 += abs(y0)
        y0 = 0

    # 8. Przycięcie i zwrot
    crop = img[y0:y1, x0:x1]

    return crop


if __name__ == "__main__":
    # Przykład użycia:
    out = crop_disk_from_image("training/original/picture22.jpg")
    if out is not None:
        cv2.imwrite("disk1_crop.jpg", out)
        # k:
        mask_save_path = "output_contours/mask_disk1.png"
        cropped_save_path = "output_contours/disk1_cropped.jpg"
        cv2.imwrite(cropped_save_path, out)
        print(f"Zapisano przycięty obraz: {cropped_save_path}")
    else:
        print("Nie znaleziono dysku na obrazie.")
