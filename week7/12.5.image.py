import cv2

image_path = "week7\lady.png"  
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(image, (200, 200))  

cv2.imshow("gray_image:", gray_image)
cv2.imshow("resized_image:", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()