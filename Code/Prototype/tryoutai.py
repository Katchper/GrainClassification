import cv2
import numpy as np

# Read the original image
original_image = cv2.imread("C:/Users/Katch/Desktop/Major Project/grain photos/broken/broken over 1_5mm001.tif")

cropped_image = original_image[1220:3400, 160:2350]

hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the brown/yellow color range
lower_bound = np.array([20, 5, 20])  # Adjust these values based on your specific color
upper_bound = np.array([40, 255, 255])

# Threshold the image to get a binary mask of the brown/yellow grains
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Find contours in the binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and draw bounding boxes around grains
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Filter out small noise (adjust the threshold as needed)
    if w * h > 300:
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result in a resizable window
cv2.namedWindow('Grain Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Grain Detection', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()