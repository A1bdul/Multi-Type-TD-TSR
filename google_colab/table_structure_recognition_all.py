import cv2
import numpy as np
import os

def recognize_structure(img, file_name):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        img_bin = 255 - img_bin

        # Detect vertical and horizontal lines
        kernel_len_ver = max(10, img.shape[0] // 50)
        kernel_len_hor = max(10, img.shape[1] // 50)
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))

        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=5)

        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        img_vh = cv2.erode(~img_vh, (2,2), iterations=2)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result_img = img.copy()
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

        # Save the result
        result_path = os.path.join(results_dir, f"{file_name}_structure_result.jpg")
        cv2.imwrite(result_path, result_img)

        return result_path

    except Exception as e:
        print(f"Error processing image {file_name}: {str(e)}")
        return None

# Renamed function
def process_table_structure(image_path):
    try:
        image = cv2.imread(image_path)
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        result_path = recognize_structure(image, file_name)
        
        if result_path:
            print(f"Processed table structure saved at: {result_path}")
        else:
            print(f"Failed to process table structure: {file_name}")
        
    except Exception as e:
        print(f"Error processing table structure in {image_path}: {str(e)}")

# Example usage
# process_table_structure("path/to/your/image.jpg")
