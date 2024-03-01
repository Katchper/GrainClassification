import cv2
import numpy as np

# Load image
original_image = cv2.imread("C:/Users/Katch/Desktop/Major Project/grain photos/groat3/21QC_270.tif", cv2.IMREAD_GRAYSCALE)
colour_image = cv2.imread("C:/Users/Katch/Desktop/Major Project/grain photos/groat3/21QC_270.tif")
cropped_image2 = colour_image[1230:3400, 160:2350]
#dst2 = cv2.resize(cropped_image2, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)

cropped_image = original_image[1230:3400, 160:2350]
#dst = cv2.resize(cropped_image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
ret,thresh1 = cv2.threshold(cropped_image,95,255,cv2.THRESH_BINARY)

blur = cv2.GaussianBlur(cropped_image,(5,5),0)
ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.filterByArea = True;
params.minArea = 100;

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(th4)

# Draw blobs on our image as red circles
im_with_keypoints = cv2.drawKeypoints(cropped_image2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(im_with_keypoints, text, (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
cv2.namedWindow('Grain Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Grain Detection', im_with_keypoints)
cv2.waitKey(0)
