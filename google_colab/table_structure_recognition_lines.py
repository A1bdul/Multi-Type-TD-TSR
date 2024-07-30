import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess

def recognize_structure(img):
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply adaptive thresholding to get binary image
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Display binary image
    plt.figure(figsize=(10, 10))
    plt.title("Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    # Invert the binary image
    img_bin = 255 - img_bin

    # Display the inverted binary image
    plt.figure(figsize=(10, 10))
    plt.title("Inverted Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    # Define kernels for vertical and horizontal line detection
    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Detect vertical lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)

    # Display vertical lines
    plt.figure(figsize=(10, 10))
    plt.title("Vertical Lines")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect horizontal lines
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)

    # Display horizontal lines
    plt.figure(figsize=(10, 10))
    plt.title("Horizontal Lines")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)

    # Display combined lines
    plt.figure(figsize=(10, 10))
    plt.title("Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Perform bitwise operations
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Display bitwise not result
    plt.figure(figsize=(10, 10))
    plt.title("Bitwise NOT")
    plt.imshow(bitnot, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect contours for box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    # Sort contours by top to bottom
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Create list of heights for detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    # Get position (x, y), width, and height for every contour
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9*img_width and h < 0.9*img_height):
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # Display the boxes on the image
    plt.figure(figsize=(10, 10))
    plt.title("Detected Boxes")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Sorting boxes into rows and columns
    row = []
    column = []
    for i in range(len(box)):
        if i == 0:
            column.append(box[i])
            previous = box[i]
        else:
            if box[i][1] <= previous[1] + mean / 2:
                column.append(box[i])
                previous = box[i]
                if i == len(box) - 1:
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    # Calculate maximum number of columns
    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        if current > countcol:
            countcol = current
            index = i

    # Retrieve the center of each column
    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    center = np.array(center)
    center.sort()

    # Arrange boxes in respective order
    finalboxes = []
    for i in range(len(row)):
        lis = [[] for _ in range(countcol)]
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin
