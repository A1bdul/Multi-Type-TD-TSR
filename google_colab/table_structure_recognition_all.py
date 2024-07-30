import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess

def recognize_structure(img):
    # Uncomment and set this path if using Tesseract locally
    # tess.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape

    # Display the grayscale image
    plt.figure(figsize=(8, 6))
    plt.title("Grayscale Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    # Thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    # Display the binary image
    plt.figure(figsize=(8, 6))
    plt.title("Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    invert = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height and 
            (w > max(10, img_width / 30) and h > max(10, img_height / 30))):
            invert = True
            img_bin[y:y+h, x:x+w] = 255 - img_bin[y:y+h, x:x+w]

    img_bin = 255 - img_bin if invert else img_bin

    # Display the binary image after inversion if needed
    plt.figure(figsize=(8, 6))
    plt.title("Inverted Binary Image")
    plt.imshow(img_bin, cmap='gray')
    plt.axis('off')
    plt.show()

    img_bin_inv = 255 - img_bin
    plt.figure(figsize=(8, 6))
    plt.title("Binary Inverted")
    plt.imshow(img_bin_inv, cmap='gray')
    plt.axis('off')
    plt.show()

    # Define kernels for vertical and horizontal line detection
    kernel_len_ver = max(10, img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Detect vertical lines
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    plt.figure(figsize=(8, 6))
    plt.title("Vertical Lines")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect horizontal lines
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)
    plt.figure(figsize=(8, 6))
    plt.title("Horizontal Lines")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    plt.figure(figsize=(8, 6))
    plt.title("Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Eroding and thresholding the combined lines image
    img_vh = cv2.dilate(img_vh, kernel, iterations=3)
    thresh, img_vh = cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(8, 6))
    plt.title("Thresholded Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine binary images
    bitor = cv2.bitwise_or(img_bin, img_vh)
    plt.figure(figsize=(8, 6))
    plt.title("Bitwise OR")
    plt.imshow(bitor, cmap='gray')
    plt.axis('off')
    plt.show()

    img_median = bitor
    plt.figure(figsize=(8, 6))
    plt.title("Median Image")
    plt.imshow(img_median, cmap='gray')
    plt.axis('off')
    plt.show()

    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, img_height*2))
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    plt.figure(figsize=(8, 6))
    plt.title("Vertical Lines Eroded")
    plt.imshow(vertical_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 3))
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    plt.figure(figsize=(8, 6))
    plt.title("Horizontal Lines Eroded")
    plt.imshow(horizontal_lines, cmap='gray')
    plt.axis('off')
    plt.show()

    # Combine final vertical and horizontal lines
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    plt.figure(figsize=(8, 6))
    plt.title("Final Combined Lines")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("Inverted Final Combined Lines")
    plt.imshow(~img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Eroding and thresholding the final image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    plt.figure(figsize=(8, 6))
    plt.title("Eroded Final Image")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()
    
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(8, 6))
    plt.title("Thresholded Final Image")
    plt.imshow(img_vh, cmap='gray')
    plt.axis('off')
    plt.show()

    # Perform XOR and NOT operations
    bitxor = cv2.bitwise_xor(img_bin, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    plt.figure(figsize=(8, 6))
    plt.title("Bitwise NOT")
    plt.imshow(bitnot, cmap='gray')
    plt.axis('off')
    plt.show()

    # Detect contours in the final image
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method in ["right-to-left", "bottom-to-top"]:
            reverse = True
        if method in ["top-to-bottom", "bottom-to-top"]:
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)

    # Initialize the image variable
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9 * img_width and h < 0.9 * img_height):
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    plt.figure(figsize=(8, 6))
    plt.title("Detected Boxes")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.axis('off')
    plt.show()

    # Creating two lists to define row and column in which cell is located
    row = []
    column = []
    j = 0

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
        lis = [[] for _ in range(countcol)]
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    return finalboxes, img_bin

# Usage example
# Load an image and call the function
# img = cv2.imread('your_image_path.jpg')
# finalboxes, img_bin = recognize_structure(img)
