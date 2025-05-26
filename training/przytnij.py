# ręczne przycinanie zdjęć i masek
import cv2
import matplotlib.pyplot as plt


nazwa="1.jpg"
x=615
y=1609
w=884
h=873

sfoto="original/"+nazwa
smask="mask/"+nazwa

foto=cv2.imread(sfoto)
mask=cv2.imread(smask)


# Rysowanie prostokątów
foto_with_rect = foto.copy()
mask_with_rect = mask.copy()
cv2.rectangle(foto_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Zielony prostokąt
cv2.rectangle(mask_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Wyświetlanie obrazów z konturami
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Zdjęcie z konturem")
plt.imshow(cv2.cvtColor(foto_with_rect, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Maska z konturem")
plt.imshow(cv2.cvtColor(mask_with_rect, cv2.COLOR_BGR2RGB))

plt.show()

foto2=foto[x:x+h, y:y+h]
mask2=mask[x:x+h, y:y+h]

plt.imshow(foto2)


cv2.imwrite(sfoto, foto2)
cv2.imwrite(smask, mask2)