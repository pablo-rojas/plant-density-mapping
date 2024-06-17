import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def rectify_image(image, tilt_angle=80, fov=55.6, image_resolution = (960, 720)):
    # Convert FOV from degrees to radians
    fov_rad = np.deg2rad(fov)
    
    # Camera intrinsic matrix (assuming the focal length is half the image width / tan(fov/2))
    f = (image_resolution[0] / 2) / np.tan(fov_rad / 2)
    K = np.array([
        [f, 0, image_resolution[0] / 2],
        [0, f, image_resolution[1] / 2],
        [0, 0, 1]
    ])
    
    # Calculate the rotation matrix for the tilt
    tilt_rad = np.deg2rad(tilt_angle)
    R = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad), np.cos(tilt_rad)]
    ])
    
    # Calculate the homography matrix from the rotation and camera matrix
    H = np.dot(K, np.dot(R, np.linalg.inv(K)))

    # Calculate the bounds of the transformed image
    height, width = image.shape[:2]
    mid_height = height // 2  # Starting height for the bottom half
    corners = np.array([
        [0, mid_height],
        [width-1, mid_height],
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.float32).reshape(1, -1, 2)  # Shape (1, 4, 2), ensure (x, y) pairs

    # Transform corners to find the extent of the new image
    transformed_corners = cv2.perspectiveTransform(corners, H)[0]
    x_coords = transformed_corners[:, 0]
    y_coords = transformed_corners[:, 1]

    # Determine new image bounds
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    # Adjust Homography to shift to positive coordinates
    translate_H = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    adjusted_H = np.dot(translate_H, H)

    # Warp image with new homography matrix
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))
    rectified_image = cv2.warpPerspective(image, adjusted_H, (new_width, new_height))

    return rectified_image

def analyse_image(image, lower_bound=np.array([33, 0, 0]), upper_bound=np.array([69, 255, 255]), display=False):

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    # Define the range of green color in HSV
    lower_all = np.array([0, 0, 0])  # Lower bound of green color
    upper_all = np.array([255, 255, 255])  # Upper bound of green color

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    domain = cv2.inRange(hsv_image, lower_all, upper_all)

    # Calculate the area of the green mask
    area = np.sum(mask == 255)

    rect_mask = rectify_image(mask)
    rect_domain = rectify_image(domain)

    # Calculate the area of the green mask
    area2 = np.sum(rect_mask == 255)

    area_domain = np.sum(domain == 255)
    area_domain2 = np.sum(rect_domain == 255)

    if (display):

        # Extract the Hue channel
        hue_channel = hsv_image[:, :, 0]

        # Calculate the histogram of the Hue channel
        # Arguments are [images], [channels], mask, [histSize (number of bins)], [range]
        hue_histogram = cv2.calcHist([hue_channel], [0], None, [255], [0, 255])
        # Plot the histogram
        plt.figure()
        plt.title("Hue Histogram")
        plt.xlabel("Hue")
        plt.ylabel("Frequency")
        plt.plot(hue_histogram)
        plt.xlim([0, 255])
        #plt.show()


        print("Area: " + str(area/area_domain))
        print("Rectified Area: " + str(area2/area_domain2))

        # Display the original and segmented images
        cv2.imshow('Original Image', image)
        cv2.imshow('Segmented Image', mask)

        # Wait for a key press and then terminate the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the original and segmented images
        # cv2.imshow('Original Image', rectified_image)
        # cv2.imshow('Segmented Image', rect_mask)

        # Wait for a key press and then terminate the windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        rect_mask = np.clip(rect_mask, 0, 1).astype(np.uint8)

    return (area2/area_domain2), rect_mask


if __name__ == "__main__":
    # Specify the path to your images directory
    directory = './imgs/'

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out the files to include only images (common formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [file for file in all_files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Read and process each image
    for image_file in image_files:
        # Construct the full image path
        image_path = os.path.join(directory, image_file)

        # Read the image
        image = cv2.imread(image_path)

        # Convert colorspace
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rectify image
        rectified_image = rectify_image(image)

        # Analyse
        coverage, mask = analyse_image(image, display=True)

        # Write rectified image
        cv2.imwrite('./rectified/' + image_file, rectified_image)







