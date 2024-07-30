import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess
import pytesseract

def recognize_structure(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # Display the grayscale image
    plt.figure(figsize=(10, 8))
    plt.title("Grayscale Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    
    # Display the binary image
    plt.figure(figsize=(10, 8))
    plt.title("Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    # Inverting the binary image
    img_bin_inv = 255 - img_bin
    plt.figure(figsize=(10, 8))
    plt.title("Inverted Binary Image")
    plt.imshow(img_bin_inv, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect vertical and horizontal lines
    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))

    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    
    plt.figure(figsize=(10, 8))
    plt.title("Vertical Lines")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    
    plt.figure(figsize=(10, 8))
    plt.title("Horizontal Lines")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    plt.figure(figsize=(10, 8))
    plt.title("Combined Vertical and Horizontal Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    img_vh = cv2.dilate(img_vh, (2, 2), iterations=5)
    plt.figure(figsize=(10, 8))
    plt.title("Dilated Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    thresh, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(10, 8))
    plt.title("Thresholded Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    bitor = cv2.bitwise_or(img_bin, img_vh)
    plt.figure(figsize=(10, 8))
    plt.title("Bitwise OR of Binary and Combined Lines")
    plt.imshow(bitor, cmap='gray')
    plt.axis('off')
    plt.show()

    img_median = cv2.medianBlur(bitor, 3)
    plt.figure(figsize=(10, 8))
    plt.title("Median Blurred Image")
    plt.imshow(img_median, cmap='gray')
    plt.axis('off')
    plt.show()

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height * 2))
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    plt.figure(figsize=(10, 8))
    plt.title("Vertical Lines After Median Blur")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width * 2, 1))
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    plt.figure(figsize=(10, 8))
    plt.title("Horizontal Lines After Median Blur")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    plt.figure(figsize=(10, 8))
    plt.title("Final Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    img_vh = cv2.erode(~img_vh, (2, 2), iterations=2)
    plt.figure(figsize=(10, 8))
    plt.title("Eroded Final Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(10, 8))
    plt.title("Thresholded Final Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    plt.figure(figsize=(10, 8))
    plt.title("Bitwise NOT of XOR Result")
    plt.imshow(bitnot, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect contours
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    plt.figure(figsize=(10, 8))
    plt.title("Detected Boxes")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    row = []
    column = []
    j = 0

    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    countcol = 0
    index = 0
    for i in range(len(row)):
        current = len(row[i])
        if current > countcol:
            countcol = current
            index = i

    center = [int(row[index][j][0] + row[index][j][2] / 2) for j in range(len(row[index]))]
    center = np.array(center)
    center.sort()

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin
