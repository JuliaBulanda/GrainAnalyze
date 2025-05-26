# ręczne przycinanie zdjęć i masek
import cv2
import matplotlib.pyplot as plt


nazwa="14.jpg"
x=500
y=1400
w=h=950
# h=873

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
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
plt.title("Zdjęcie z konturem")
plt.imshow(cv2.cvtColor(foto_with_rect, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 2, 2)
# plt.title("Maska z konturem")
# plt.imshow(cv2.cvtColor(mask_with_rect, cv2.COLOR_BGR2RGB))

plt.show()

foto2 = foto[y:y+h, x:x+w]
mask2 = mask[y:y+h, x:x+w]


plt.imshow(foto2)


cv2.imwrite(sfoto, foto2)
cv2.imwrite(smask, mask2)