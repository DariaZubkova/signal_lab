from imageio import imread, imsave
from skimage.color import rgb2gray, rgba2rgb
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_opening, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import label2rgb
from skimage import measure


def find_size(contour, contours, mask_draw, number):
    max_x = contour[0]
    max_y = contour[0]
    min_x = contour[0]
    min_y = contour[0]
    for i in range(len(contour)):
        pair = contour[i]
        if max_x[1] < pair[1]:
            max_x = pair
        if min_x[1] > pair[1]:
            min_x = pair
        if min_y[0] < pair[0]:
            min_y = pair
        if max_y[0] > pair[0]:
            max_y = pair

    print("blue = ", max_x)
    print("yellow = ", min_x)
    print("red = ", max_y)
    print("green = ", min_y)

    fig, ax = plt.subplots()
    ax.imshow(mask_draw, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.plot(max_x[1], max_x[0], 'bo')  # max_y --> самая правая (max_x)
    ax.plot(min_x[1], min_x[0], 'yo')  # min_y --> самая левая (min_x)
    ax.plot(max_y[1], max_y[0], 'ro')  # min_x --> самая верхняя (max_y)
    ax.plot(min_y[1], min_y[0], 'go')  # max_x --> самая нижнаяя (min_y)

    if number == 1:
        # weidth = max_x[1] - min_x[1]   #b - y (blue - red?)
        weidth = max_x[1] - min_y[1]  # по х    blue - green
        # height = min_y[0] - min_x[0]   #g - y
        # height = min_y[0] - max_y[0]   #g - r
        height = max_x[0] - max_y[0]  # по у b - r
    else:
        weidth = max_x[1] - min_y[1]
        height = min_y[0] - min_x[0]  # по у
    return weidth, height, max_x, max_y, min_x, min_y

def check(Weidth_A, Height_A, Weidht_B, Height_B):
    print("Check:")
    if Weidth_A >= Weidht_B:
        print("Weidth A >= weidth B")
    else:
        print("WARNING: Weidth A < weidth B")
    if Height_A >= Height_B:
        print("Height A >= height B")
    else:
        print("WARNING: Height A < height B")

####### Part 1 - Object B: find contour
name_file = './dataset/data1.jpg' #warning: 5,7,10

img = imread(name_file)
img2 = rgb2gray(img)

binary_img = binary_closing(canny(img2, sigma=2), selem=np.ones((6, 6))) #1.5;2
img_segment = binary_fill_holes(binary_img)
mask_img = binary_opening(img_segment, selem=np.ones((40, 40)))#16;25

plt.imshow(label2rgb(mask_img, image=img2))

# Find contours
contours_img = measure.find_contours(mask_img, 0.8)

# Display
fig, ax = plt.subplots()
ax.imshow(mask_img, cmap=plt.cm.gray)

for contour in contours_img:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

plt.show()

print(len(contours_img))

####### Part 2 - Object B: find dots and small box's size
if len(contours_img) == 1:
    Weidth_B, Height_B, max_x_B, max_y_B, min_x_B, min_y_B= find_size(contours_img[0], contours_img, mask_img, 1)
    print(Weidth_B)
    print(Height_B)
else:
    print('ERROR')
    Weidth_B = 0
    Height_B = 0

####### Part 3 - Object A: find contours on marks
img_pink = cv2.imread(name_file)

PINK_MIN = np.array([145, 65, 65],np.uint8)
PINK_MAX = np.array([179, 255, 255],np.uint8)

hsv_img = cv2.cvtColor(img_pink,cv2.COLOR_BGR2HSV)

pink_markers = cv2.inRange(hsv_img, PINK_MIN, PINK_MAX)

mask_pink = binary_opening(pink_markers, selem=np.ones((10, 10)))

plt.imshow(mask_pink)

# Find contours
contours_pink = measure.find_contours(mask_pink, 0.8)

# Display
fig, ax = plt.subplots()
ax.imshow(pink_markers, cmap=plt.cm.gray)

for contour in contours_pink:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

plt.show()

print(len(contours_pink))

####### Part 4 - Object A: find dots and big box's size
if len(contours_pink) == 2:
    Weidth_A1, Height_A1, max_x_A1, max_y_A1, min_x_A1, min_y_A1 = find_size(contours_pink[0], contours_pink, mask_pink, 2)
    Weidtht_A2, Height_A2, max_x_A2, max_y_A2, min_x_A2, min_y_A2 = find_size(contours_pink[1], contours_pink, mask_pink, 2)
    d_11 = np.sqrt((max_x_A1[1] - max_y_A1[1])**2 + (max_x_A1[0] - max_y_A1[0])**2) #по у правильнее!!!

    Weidth_A = max_x_A1[1] - min_y_A2[1] #по х
    if Height_A1 > Height_A2:
        Height_A = Height_A1
    else:
        Height_A = Height_A2

    print(Weidth_A)
    print(Height_A)
else:
    print('ERROR')
    Weidth_A = 0
    Height_A = 0

####### Part 5 - Check: check of sizes
if Weidth_A != 0 and Height_A != 0 and Weidth_B != 0 and Height_B != 0:
    check(Weidth_A, Height_A, Weidth_B, Height_B)
else:
    print("ERROR")
