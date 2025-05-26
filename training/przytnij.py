# ręczne przycinanie zdjęć i masek
import cv2

nazwa="1.jpg"
x=615
y=1609
w=884
h=873

foto=cv2.imread("original/"+nazwa)
mask=cv2.imread("mask/"+nazwa)

# print(foto)
#
# cv2.imshow("1", foto)
# cv2.imshow("2", mask)
# cv2.waitKey()

foto2=foto[x:x+h][y:y+h]
mask2=mask[x:x+h][y:y+h]

cv2.imwrite("orginal/"+nazwa, foto2)
cv2.imwrite("mask/"+nazwa, mask2)