# ręczne przycinanie zdjęć i masek
import cv2

nazwa="1.jpg"
x=615
y=01609
w=884
h=0873

foto=cv2.imread("orginal/"+nazwa)
mask=cv2.imread("mask/"+nazwa)

foto2=foto[x:x+h][y:y+h]
mask2=mask[x:x+h][y:y+h]

cv2.imwrite("orginal/"+nazwa, foto2)
cv2.imwrite("mask/"+nazwa, mask2)