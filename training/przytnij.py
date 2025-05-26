# ręczne przycinanie zdjęć i masek
import cv2
import matplotlib.pyplot as plt


nazwa="15.jpg"
# x=430
# y=1480
# w=h=900
# # h=873

sfoto="original/"+nazwa
smask="mask/"+nazwa

foto=cv2.imread(sfoto)
mask=cv2.imread(smask)

# wymiary oryginałów
h_img, w_img = foto.shape[:2]

# bok kwadratu jako mniejszy z wymiarów
side = min(h_img, w_img)

# lewy-górny róg przycięcia (centrowanie)
x = (w_img - side) // 2
y = (h_img - side) // 2



# Rysowanie prostokątów
foto_with_rect = foto.copy()
mask_with_rect = mask.copy()
cv2.rectangle(foto_with_rect, (x, y), (x + side, y + side), (0, 255, 0), 2)  # Zielony prostokąt
cv2.rectangle(mask_with_rect, (x, y), (x + side, y + side), (0, 255, 0), 2)

# Wyświetlanie obrazów z konturami
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
plt.title("Zdjęcie z konturem")
plt.imshow(cv2.cvtColor(foto_with_rect, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 2, 2)
# plt.title("Maska z konturem")
# plt.imshow(cv2.cvtColor(mask_with_rect, cv2.COLOR_BGR2RGB))

plt.show()

# foto2 = foto[y:y+h, x:x+w]
# mask2 = mask[y:y+h, x:x+w]

# przycinanie — tu zachowujemy foto2 i mask2
foto2 = foto[y:y+side, x:x+side]
mask2 = mask[y:y+side, x:x+side]

plt.imshow(foto2)


cv2.imwrite(sfoto, foto2)
cv2.imwrite(smask, mask2)