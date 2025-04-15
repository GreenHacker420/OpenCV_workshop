#pip install opencv-python opencv-contrib-python numpy pyautogui matplotlib scipy pillow cmake dlib 

# import numpy as np
# import cv2

# # Create a blank 300x300 black image (3 channels for color)
# image = np.zeros((300, 300, 3), dtype=np.uint8)

# # Set specific pixel colors
# image[50:100, 50:100] = [255, 0, 0]   # Blue square
# image[100:150, 100:150] = [0, 255, 0] # Green square
# image[150:200, 150:200] = [0, 0, 255] # Red square
# image[200:250, 200:250] = [255, 255, 255] # White square

# # Show the image
# cv2.imshow("Image as Matrix", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Optional: Save the image
# cv2.imwrite("color_blocks.png", image)


# import numpy as np
# import cv2

# # Create a 400x400 black image with 3 channels (BGR)
# image = np.zeros((400, 400, 3), dtype=np.uint8)

# # Define block size and color mapping
# block_size = 100
# colors = {
#     'Blue': (255, 0, 0),
#     'Green': (0, 255, 0),
#     'Red': (0, 0, 255),
#     'White': (255, 255, 255)
# }

# # Draw 4 colored blocks with labels
# positions = [(0, 0), (0, 100), (100, 0), (100, 100)]
# color_names = list(colors.keys())

# for (x, y), name in zip(positions, color_names):
#     color = colors[name]
#     cv2.rectangle(image, (x, y), (x + block_size, y + block_size), color, -1)
#     cv2.putText(image, name, (x + 10, y + 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if name == "White" else (255, 255, 255), 2)

# # Show image
# cv2.imshow("Image Matrix Visualization", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# # Create a black image of size 200x200 pixels
# image = np.zeros((200, 200, 3), dtype=np.uint8)

# # Add a blue square at top-left
# image[0:100, 0:100] = [255, 0, 0]  # Blue in BGR

# # Add a green square at top-right
# image[0:100, 100:200] = [0, 255, 0]  # Green

# # Add a red square at bottom-left
# image[100:200, 0:100] = [0, 0, 255]  # Red

# # Add a white square at bottom-right
# image[100:200, 100:200] = [255, 255, 255]  # White

# # Show the image
# cv2.imshow("Color Blocks", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np

# Create a black image of size 200x200 pixels
image = np.zeros((200, 200, 3), dtype=np.uint8)

# Add colored squares
image[0:100, 0:100] = [255, 0, 0]      # Blue
image[0:100, 100:200] = [0, 255, 0]    # Green
image[100:200, 0:100] = [0, 0, 255]    # Red
image[100:200, 100:200] = [255, 255, 255]  # White

# --- Draw over the image ---

# 1. Add rectangles inside each block (with padding)
cv2.rectangle(image, (10, 10), (90, 90), (0, 0, 0), 2)           # Top-left (Blue)
cv2.rectangle(image, (110, 10), (190, 90), (0, 0, 0), 2)         # Top-right (Green)
cv2.rectangle(image, (10, 110), (90, 190), (255, 255, 255), 2)   # Bottom-left (Red)
cv2.rectangle(image, (110, 110), (190, 190), (0, 0, 0), 2)       # Bottom-right (White)

# 2. Add center circle
cv2.circle(image, (100, 100), 10, (255, 255, 0), -1)  # Yellow dot at center

# 3. Add cross lines (diagonals)
# cv2.line(image, (0, 0), (200, 200), (255, 255, 0), 1)
# cv2.line(image, (200, 0), (0, 200), (255, 255, 0), 1)

# 4. Add text labels to each block
cv2.putText(image, "Blue", (10, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0), 1)
cv2.putText(image, "Green", (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
cv2.putText(image, "Red", (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
cv2.putText(image, "White", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# Show the final image
cv2.imshow("Annotated Blocks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 1. Virtual Paint App (Mouse Drawing with Shape Options)
# # Enhanced Virtual Paint App with UI, Shape Tools, Color Picker, Brush Size, and Preview
# import cv2
# import numpy as np
# import pyautogui

# # Get screen size dynamically
# screen_width, screen_height = pyautogui.size()

# # Constants
# UI_HEIGHT = 50
# CANVAS_WIDTH = screen_width
# CANVAS_HEIGHT = screen_height - UI_HEIGHT
# WINDOW_HEIGHT = CANVAS_HEIGHT + UI_HEIGHT

# # Canvas
# paint_window = np.ones((WINDOW_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255

# # Drawing state
# is_drawing = False
# start_point = (-1, -1)
# last_point = (-1, -1)
# current_shape = 'circle'

# # Drawing settings
# circle_color = (0, 0, 255)
# rectangle_color = (255, 0, 0)
# line_color = (0, 255, 0)
# freestyle_color = (0, 0, 0)
# eraser_color = (255, 255, 255)
# eraser_size = 15
# freestyle_thickness = 2

# # Color picker (preset colors)
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (255, 255, 255)]
# selected_color = freestyle_color


# def create_ui_panel():
#     button_width = CANVAS_WIDTH // 6
#     paint_window[0:UI_HEIGHT, :] = (200, 200, 200)

#     # Shape Buttons
#     cv2.rectangle(paint_window, (0, 0), (button_width, UI_HEIGHT), (220, 220, 220), -1)
#     cv2.putText(paint_window, "Circle", (10, UI_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#     cv2.rectangle(paint_window, (button_width, 0), (button_width*2, UI_HEIGHT), (220, 220, 220), -1)
#     cv2.putText(paint_window, "Rect", (button_width+10, UI_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#     cv2.rectangle(paint_window, (button_width*2, 0), (button_width*3, UI_HEIGHT), (220, 220, 220), -1)
#     cv2.putText(paint_window, "Line", (button_width*2+10, UI_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#     cv2.rectangle(paint_window, (button_width*3, 0), (button_width*4, UI_HEIGHT), (220, 220, 220), -1)
#     cv2.putText(paint_window, "Draw", (button_width*3+10, UI_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#     cv2.rectangle(paint_window, (button_width*4, 0), (button_width*5, UI_HEIGHT), (220, 220, 220), -1)
#     cv2.putText(paint_window, "Eraser", (button_width*4+10, UI_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#     # Color Picker
#     for i, color in enumerate(colors):
#         x1 = button_width*5 + i*30
#         x2 = x1 + 20
#         cv2.rectangle(paint_window, (x1, 5), (x2, UI_HEIGHT-5), color, -1)
#         if color == selected_color:
#             cv2.rectangle(paint_window, (x1, 5), (x2, UI_HEIGHT-5), (0, 0, 0), 2)


# def draw(event, x, y, flags, param):
#     global is_drawing, start_point, last_point, current_shape, paint_window, selected_color, freestyle_thickness, eraser_size

#     y_canvas = y - UI_HEIGHT
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if y < UI_HEIGHT:
#             button_width = CANVAS_WIDTH // 6
#             if x < button_width:
#                 current_shape = 'circle'
#             elif x < button_width * 2:
#                 current_shape = 'rectangle'
#             elif x < button_width * 3:
#                 current_shape = 'line'
#             elif x < button_width * 4:
#                 current_shape = 'freestyle'
#             elif x < button_width * 5:
#                 current_shape = 'eraser'
#             else:
#                 index = (x - button_width*5) // 30
#                 if 0 <= index < len(colors):
#                     selected_color = colors[index]
#                     freestyle_color = selected_color
#             create_ui_panel()
#             return

#         is_drawing = True
#         start_point = (x, y_canvas)
#         last_point = (x, y_canvas)

#     elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
#         temp_window = paint_window.copy()
#         current_point = (x, y_canvas)

#         if current_shape == 'circle':
#             radius = int(np.linalg.norm(np.array(start_point) - np.array(current_point)))
#             cv2.circle(temp_window[UI_HEIGHT:], start_point, radius, selected_color, -1)

#         elif current_shape == 'rectangle':
#             cv2.rectangle(temp_window[UI_HEIGHT:], start_point, current_point, selected_color, 2)

#         elif current_shape == 'line':
#             cv2.line(temp_window[UI_HEIGHT:], start_point, current_point, selected_color, 2)

#         elif current_shape == 'freestyle':
#             cv2.line(paint_window[UI_HEIGHT:], last_point, current_point, freestyle_color, freestyle_thickness)
#             last_point = current_point
#             temp_window = paint_window.copy()

#         elif current_shape == 'eraser':
#             cv2.circle(paint_window[UI_HEIGHT:], current_point, eraser_size, eraser_color, -1)
#             temp_window = paint_window.copy()
#             cv2.circle(temp_window[UI_HEIGHT:], current_point, eraser_size, (200, 200, 200), 1)

#         cv2.imshow("Paint", temp_window)

#     elif event == cv2.EVENT_LBUTTONUP and is_drawing:
#         is_drawing = False
#         end_point = (x, y_canvas)
#         if current_shape == 'circle':
#             radius = int(np.linalg.norm(np.array(start_point) - np.array(end_point)))
#             cv2.circle(paint_window[UI_HEIGHT:], start_point, radius, selected_color, -1)
#         elif current_shape == 'rectangle':
#             cv2.rectangle(paint_window[UI_HEIGHT:], start_point, end_point, selected_color, 2)
#         elif current_shape == 'line':
#             cv2.line(paint_window[UI_HEIGHT:], start_point, end_point, selected_color, 2)


# # Setup
# cv2.namedWindow("Paint", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Paint", CANVAS_WIDTH, WINDOW_HEIGHT)
# create_ui_panel()
# cv2.setMouseCallback("Paint", draw)

# print("Controls: 'c'=circle, 'r'=rectangle, 'l'=line, 'f'=freestyle, 'e'=eraser, '+'/'-'=brush size, 's'=save, 'q'=quit")

# while True:
#     cv2.imshow("Paint", paint_window)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('c'):
#         current_shape = 'circle'
#     elif key == ord('r'):
#         current_shape = 'rectangle'
#     elif key == ord('l'):
#         current_shape = 'line'
#     elif key == ord('f'):
#         current_shape = 'freestyle'
#     elif key == ord('e'):
#         current_shape = 'eraser'
#     elif key == ord('+'):
#         freestyle_thickness = min(freestyle_thickness + 1, 20)
#         eraser_size = min(eraser_size + 1, 50)
#     elif key == ord('-'):
#         freestyle_thickness = max(freestyle_thickness - 1, 1)
#         eraser_size = max(eraser_size - 1, 5)
#     elif key == ord('s'):
#         cv2.imwrite("drawing_output.png", paint_window[UI_HEIGHT:])
#         print("Saved as drawing_output.png")
#     create_ui_panel()

# cv2.destroyAllWindows()







# import cv2 as cv
# import numpy as np

# image = cv.imread('logo.jpg')
# print(image)
# print("Shape",image.shape)
# print("Size",image.size)
# print("Data Type",image.dtype)

# resized_image = cv.resize(image , (200 , 200))
# print()
# grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# gaussian_blurred_image = cv.GaussianBlur(image, (15, 15), 0)
# edges = cv.Canny(image, 100, 200)



# while True:
#     # Display the image
#     cv.imshow('Original Image', image)

#     # Wait for a key press
#     key = cv.waitKey(1) & 0xFF

#     # If 'q' is pressed, exit the loop
#     if key == ord('q'):
#         break


# while True:
#     #using video capture
#     cap = cv.VideoCapture(0)
#     # Check if the camera opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         break
#     # Read a frame from the camera
#     ret, frame = cap.read()
#     # Check if the frame was read successfully
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#     # Display the frame
#     cv.imshow('Camera Feed', frame)

#     # Wait for a key press
#     key = cv.waitKey(1) & 0xFF
#     # If 'q' is pressed, exit the loop
#     if key == ord('q'):
#         break
# # Release the camera and close all OpenCV windows
# cap.release()
