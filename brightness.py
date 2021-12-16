import cv2

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

original = cv2.imread(r"C:\Users\Ranajoy\Downloads\CASIA-WebFace\CASIA-WebFace\0000099\010.jpg")
frame = increase_brightness(original, value=90)

cv2.imshow("Image", original)
cv2.imshow("Detail_Enhance", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()