import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess

def recognize_structure(img):
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # Display the grayscale image
    plt.figure(figsize=(10, 10))
    plt.title("Grayscale Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply adaptive thresholding to get binary image
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

    # Display binary image
    plt.figure(figsize=(10, 10))
    plt.title("Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply median blur to reduce noise
    img_median = cv2.medianBlur(img_bin, 3)

    # Display median blurred image
    plt.figure(figsize=(10, 10))
    plt.title("Median Blurred Image")
    plt.imshow(img_median, cmap='gray')
    plt.axis('off')
    plt.show()

    # Define kernels for vertical and horizontal line detection
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height * 2))
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)

    # Display vertical lines
    plt.figure(figsize=(10, 10))
    plt.title("Vertical Lines")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width * 2, 9))
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)

    # Display horizontal lines
    plt.figure(figsize=(10, 10))
    plt.title("Horizontal Lines")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine horizontal and vertical lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    # Display combined lines
    plt.figure(figsize=(10, 10))
    plt.title("Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    img_vh_inverted = ~img_vh
    # Display inverted combined lines
    plt.figure(figsize=(10, 10))
    plt.title("Inverted Combined Lines")
    plt.imshow(img_vh_inverted, cmap='gray')
    plt.axis('off')
    plt.show()

    # Erode and threshold the combined lines image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_vh = cv2.erode(img_vh_inverted, kernel, iterations=2)
    _, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)

    # Display the thresholded image
    plt.figure(figsize=(10, 10))
    plt.title("Thresholded Image")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Perform bitwise XOR and NOT operations
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Display the result of bitwise NOT operation
    plt.figure(figsize=(10, 10))
    plt.title("Bitwise NOT Result")
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

    # Create a list of heights for detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    # Create list box to store all boxes
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    # Display the boxes on the image
    plt.figure(figsize=(10, 10))
    plt.title("Detected Boxes")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Sorting the boxes into rows and columns
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
