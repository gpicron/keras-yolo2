import cv2

print(cv2.ocl.haveOpenCL())
print(cv2.ocl.useOpenCL())
cv2.ocl.setUseOpenCL(True)
print(cv2.ocl.useOpenCL())

img = cv2.UMat(cv2.imread("images/cat.jpg", cv2.IMREAD_COLOR))
imgUMat = cv2.UMat(img)
gray = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
gray = cv2.Canny(gray, 0, 50)

cv2.imshow("edges", gray)
cv2.waitKey();
