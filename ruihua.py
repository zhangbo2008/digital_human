import cv2
a=cv2.imread('debug0原始128.png')
b=cv2.resize(a,(250,260),interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('fffffffff.png',b)
b=cv2.resize(a,(250,260),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('fffffffff1.png',b)
b=cv2.resize(a,(250,260),interpolation=cv2.INTER_AREA)
cv2.imwrite('fffffffff2.png',b)
print("可以看到resize函数cv2.INTER_CUBIC 和 cv2.INTER_LANCZOS4 效果最好, 看不出任何差别.")