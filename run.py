import os
import re
from utils import rectify_image, analyse_image
import cv2
import numpy as np


import cv2
from djitellopy import Tello

####################################################################################################
#                                             CONSTANTS                                            #
####################################################################################################

# Grid shape
NUM_LINES = 3
NUM_MOVES = 5

# Movement parameters
MOVE_DISTANCE = 400
START_DISTANCE = 10
LENGTH = 440
ROTATE_ANGLE = 90

# Path to the folder that will contain the images
IMG_PATH = './imgs/'

# Conversion factor cm to pix
FACTOR = 3679/LENGTH

# List all files in the folder
files = os.listdir(IMG_PATH)

# Define the pattern for matching the filename structure
filename_pattern = r"grid_frame_(\d+)_(\d+).png"

stitched_images = []

offset = (LENGTH + START_DISTANCE) * FACTOR
max_x = int(offset + (START_DISTANCE + MOVE_DISTANCE*(NUM_MOVES+1))*FACTOR)
max_y = int((NUM_LINES+1)*FACTOR*MOVE_DISTANCE)

####################################################################################################
#                                             FUNCTIONS                                            #
####################################################################################################

def rotation(line):
    if line % 2 == 0:
        tello.rotate_clockwise(ROTATE_ANGLE)
        tello.move_forward(MOVE_DISTANCE)
        tello.rotate_clockwise(ROTATE_ANGLE)
    else:
        tello.rotate_counter_clockwise(ROTATE_ANGLE)
        tello.move_forward(MOVE_DISTANCE)
        tello.rotate_counter_clockwise(ROTATE_ANGLE)

####################################################################################################
#                                             MOVEMENT                                             #
####################################################################################################

# Set up connection
tello = Tello()
tello.connect()
print("Connected!")

# Set up camera
tello.streamon()
frame_read = tello.get_frame_read()
print("Camera ready!")

# Take off
tello.takeoff()
print("Take off!")

# Grid movement loop
for line in range(1, NUM_LINES + 1):

    # Line movemnets
    for move in range(1, NUM_MOVES + 1):
        cv2.imwrite(IMG_PATH + f"grid_frame_{line}_{move}.png", frame_read.frame)
        tello.move_forward(MOVE_DISTANCE)

    # Rotate after every finished line except last
    if line < NUM_LINES:
        rotation(line)

# Landing drone
tello.land()
print("Lended!")

####################################################################################################
#                                             PROCESSING                                           #
####################################################################################################

# Iterate over each image file
for image_file in files:
    # Check if the file is an image file and matches the expected pattern
    if image_file.lower().endswith(".png") and re.match(filename_pattern, image_file):
        # Extract line and move from the filename using regular expressions
        match = re.match(filename_pattern, image_file)
        line = int(match.group(1))
        move = int(match.group(2))
        
        # Print the values of line and move
        print(f"Image file: {image_file}, Line: {line}, Move: {move}")
        
        # Construct the full path to the image file
        image_path = os.path.join(IMG_PATH, image_file)

        # Read the image
        image = cv2.imread(image_path)
        
        # Convert colorspace
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join('./rgb/', image_file), image)

        # Rectify image
        rectified_image = rectify_image(image)

        # Analyse
        coverage, mask = analyse_image(image, display=True)
        y_pos = (line-1)*MOVE_DISTANCE
        if (line % 2 == 1):
            x_pos = (move-1)*MOVE_DISTANCE + START_DISTANCE
            mask = cv2.rotate(mask, cv2.ROTATE_180)
        else:
            x_pos = (NUM_MOVES-move)*MOVE_DISTANCE - START_DISTANCE - LENGTH

        # Append the image and its position to the list
        stitched_images.append((mask, (int(x_pos*FACTOR+offset), int(y_pos*FACTOR))))

# Create a large canvas
canvas = np.zeros((max_x, max_y), dtype=np.uint8)

# Stitch images onto the canvas
for img, (x_pos, y_pos) in stitched_images:
    canvas[x_pos:x_pos+img.shape[0], y_pos:y_pos+img.shape[1]] = img

# Save or display the stitched image
cv2.imwrite('stitched_output.png', canvas)


# Prepare a large canvas as before
canvas = np.zeros((max_x, max_y), dtype=np.uint8)


# Add images onto the canvas
for img, (x_pos, y_pos) in stitched_images:
    canvas[x_pos:x_pos+img.shape[0], y_pos:y_pos+img.shape[1]] += img

# Normalize the result if needed and convert back to uint8
canvas = np.clip(canvas, 0, 255).astype(np.uint8)

# Factor by which the image resolution will be reduced
factor = 40  # Change this factor as needed

# Calculate the new dimensions
new_width = int(canvas.shape[1] / factor)
new_height = int(canvas.shape[0] / factor)

# Resize the image
resized_image = cv2.resize(canvas, (new_width, new_height), interpolation=cv2.INTER_AREA)

resized_image = cv2.GaussianBlur(resized_image, (9, 9), 5)


# Save or display the stitched image
cv2.imwrite('stitched_output2.png', resized_image)

# Define a color LUT for visualizing the mask
# Create an empty LUT with 256 entries (for 256 different grayscale levels)
lut = np.zeros((256, 1, 3), dtype=np.uint8)

# Example: Mapping grayscale values to a heatmap-style color scheme
# Blue (low intensity) to Red (high intensity)
for i in range(256):
    if i < 128:
        lut[i, 0, :] = [255 - 2*i, 2*i, 0]  # from Blue to Green
    else:
        lut[i, 0, :] = [0, 255 - 2*(i - 128), 2*(i - 128)]  # from Green to Red

# Expand the gray image to 3 channels
expanded_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

print(expanded_image.shape)
# Apply the LUT to the blurred mask
colorized_mask = cv2.LUT(expanded_image, lut)

# Save or display the result
cv2.imwrite('colorized_mask.png', colorized_mask)
cv2.imshow('Colorized Mask', colorized_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()