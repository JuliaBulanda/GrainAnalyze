import numpy as np
import tensorflow as tf
import cv2

# Załaduj model UNet
model = tf.keras.models.load_model('unet_disc_segmentation.keras')

def crop_disk_from_image(img_path):
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
    target_size = (512, 512)
    resized = cv2.resize(img, target_size)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # (1,H,W,3)

    # 3. Predykcja maski
    pred = model.predict(tensor)[0, ..., 0]
    mask = (pred > 0.5).astype(np.uint8) * 255

    # 4. Przywrócenie do pełnej rozdzielczości
    mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 5. Detekcja konturów
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # lub rzucamy wyjątek / zwracamy cały obraz

    # 6. Wybór największego konturu
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # 7. Rozmiar kwadratu (największy z boków)
    side = max(cw, ch)
    # wyśrodkowanie
    cx, cy = x + cw//2, y + ch//2
    x0 = max(0, cx - side//2)
    y0 = max(0, cy - side//2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)

    # 8. Przycięcie i zwrot
    crop = img[y0:y1, x0:x1]
    return crop


if __name__ == "__main__":
    # Przykład użycia:
    out = crop_disk_from_image("input_unet/cw/Kotlina/Samsung Galaxy A52/2.jpg")
    if out is not None:
        cv2.imwrite("disk1_crop.jpg", out)
    else:
        print("Nie znaleziono dysku na obrazie.")
