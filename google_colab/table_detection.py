import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction(img, predictor):
    outputs = predictor(img)

    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2

    img_copy = np.array(img, copy=True)  # Make a copy of the image for drawing

    for x1, y1, x2, y2 in outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy():
        # Start coordinate (top-left corner)
        start_point = int(x1), int(y1)
        # Ending coordinate (bottom-right corner)
        end_point = int(x2), int(y2)
        # Draw a rectangle with blue line borders of thickness 2 px
        cv2.rectangle(img_copy, start_point, end_point, color, thickness)

    # Displaying the image using matplotlib
    print("TABLE DETECTION:")
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.title("Detection")
    plt.axis('off')  # Hide axis
    plt.show()

def make_prediction(img, predictor):
    outputs = predictor(img)

    table_list = []
    table_coords = []

    for i, box in enumerate(outputs["instances"].get_fields()["pred_boxes"].tensor.to("cpu").numpy()):
        x1, y1, x2, y2 = box
        table_img = np.array(img[int(y1):int(y2), int(x1):int(x2)], copy=True)
        table_list.append(table_img)
        table_coords.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
        print("TABLE", i, ":")
        # Display the table image using matplotlib
        plt.imshow(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
        plt.title(f"Table {i}")
        plt.axis('off')  # Hide axis
        plt.show()
        print()

    return table_list, table_coords
