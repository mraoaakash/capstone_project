import cv2
import matplotlib.pyplot as plt



imagefile = 'dataset/sample-data/sample_blob.png'
outpath = 'dataset/input/samples'
img = cv2.imread(imagefile)

# Convert you image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Threshold the image to extract only objects that are not black
# You need to use a one channel image, that's why the slice to get the first layer
tv, thresh = cv2.threshold(gray[:,:,0], 1, 255, cv2.THRESH_BINARY)

# Get the contours from your thresholded image
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Create a copy of the original image to display the output rectangles
output = img.copy()

# Loop through your contours calculating the bounding rectangles and plotting them
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output, (x,y), (x+w, y+h), (0, 0, 255), 2)
    print("x: {}, y: {}, w: {}, h: {}".format(x, y, w, h))
# Display the output image
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig(outpath + '/sample_blob.png')