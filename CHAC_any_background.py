import cv2 as cv
import numpy as np

# read image
input_image = cv.imread("Path_name\\origin.jpg")
height, width, channels = input_image.shape

# Set Background 
background_white = np.zeros([height,width,channels],dtype=np.uint8)
background_white.fill(255)
background_image = cv.imread("Path_name\\background.jpg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", input_image)

# gray scale
gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
# two value
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# get structring elements
k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
# open operation
binary = cv.morphologyEx(binary, cv.MORPH_OPEN, k)

from skimage.segmentation import clear_border
binary = clear_border(binary, buffer_size=1) #Remove edge touching grains

cv.imshow("binary", binary)

# find contours
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# count convex contours and total contours
convex_contours = 0
total_contours = 0

for c in range(len(contours)):
    
    # using approximate shape to replace ordinary shape
    perimeter = cv.arcLength(contours[c], True)
    approximatedShape = cv.approxPolyDP(contours[c], 0.02 * perimeter, True)
    
    # convex or not
    ret = cv.isContourConvex(approximatedShape)
    # convex detect
    if ret  == True:
        points = cv.convexHull(approximatedShape)
        total = len(points)
        for i in range(len(points)):
            x1, y1 = points[i % total][0]
            x2, y2 = points[(i+1) % total][0]
            #cv.circle(input_image, (x1, y1), 4, (255, 0, 0), 2, 8, 0)
            cv.line(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
        print(points)
        print("convex : ", ret)
        convex_contours += 1
        total_contours += 1
        
    else:
        print("convex : ", ret)
        total_contours += 1

# show results
print("convex contours = " + str(convex_contours))
print("total contours = " + str(total_contours))
cv.imshow("contours_analysis", input_image)
cv.imwrite("Path_name\\output.jpg", input_image)
cv.waitKey(0)
cv.destroyAllWindows()