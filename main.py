"""
Required dependencies
python3 -m venv venv
source venv/bin/activate
brew install cmake
pip install opencv-python opencv-contrib-python numpy pyautogui matplotlib scipy pillow cmake dlib
"""

import cv2
import numpy as np

def basic_image_creation():
    """
    Demonstrates basic image creation and manipulation.
    Creates a 300x300 image with colored squares.
    """
    # Create a blank 300x300 black image (3 channels for color)
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    # Set specific pixel colors
    image[50:100, 50:100] = [255, 0, 0]   # Blue square
    image[100:150, 100:150] = [0, 255, 0] # Green square
    image[150:200, 150:200] = [0, 0, 255] # Red square
    image[200:250, 200:250] = [255, 255, 255] # White square

    # Show and save the image
    cv2.imshow("Basic Image Creation", image)
    cv2.waitKey(0)
    cv2.imwrite("basic_image.png", image)
    cv2.destroyAllWindows()

def color_blocks_with_labels():
    """
    Creates a 400x400 image with labeled color blocks.
    Demonstrates text and shape drawing.
    """
    # Create a 400x400 black image with 3 channels (BGR)
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Define block size and color mapping
    block_size = 100
    colors = {
        'Blue': (255, 0, 0),
        'Green': (0, 255, 0),
        'Red': (0, 0, 255),
        'White': (255, 255, 255)
    }

    # Draw 4 colored blocks with labels
    positions = [(0, 0), (0, 100), (100, 0), (100, 100)]
    color_names = list(colors.keys())

    for (x, y), name in zip(positions, color_names):
        color = colors[name]
        cv2.rectangle(image, (x, y), (x + block_size, y + block_size), color, -1)
        cv2.putText(image, name, (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if name == "White" else (255, 255, 255), 2)

    # Show and save the image
    cv2.imshow("Color Blocks with Labels", image)
    cv2.waitKey(0)
    cv2.imwrite("color_blocks.png", image)
    cv2.destroyAllWindows()

def resize_image(image, width=None, height=None, scale=None):
    """
    Resize an image while maintaining aspect ratio.
    """
    if scale is not None:
        return cv2.resize(image, None, fx=scale, fy=scale)
    
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / h
        dim = (int(w * ratio), height)
    else:
        ratio = width / w
        dim = (width, int(h * ratio))
    
    return cv2.resize(image, dim)

def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_blur(image, kernel_size=(5,5)):
    """
    Apply Gaussian blur to the image.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image, threshold1=100, threshold2=200):
    """
    Detect edges using Canny edge detection.
    """
    return cv2.Canny(image, threshold1, threshold2)

def apply_threshold(image, threshold=127, max_value=255, type=cv2.THRESH_BINARY):
    """
    Apply thresholding to the image.
    """
    return cv2.threshold(image, threshold, max_value, type)[1]




def image_processing_demo(image_path):
    """
    Demonstrates various image processing operations.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Get image properties
    print("\nImage Properties:")
    print(f"Shape: {image.shape}")
    print(f"Size: {image.size}")
    print(f"Data Type: {image.dtype}")

    # Apply various image processing operations
    processed_images = {
        'Original': image,
        'Resized (50%)': resize_image(image, scale=0.5),
        'Grayscale': convert_to_grayscale(image),
        'Blurred': apply_blur(image),
        'Edges': detect_edges(image),
        'Threshold': apply_threshold(convert_to_grayscale(image))
    }

    # Display all processed images
    for title, img in processed_images.items():
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return processed_images

def camera_feed_demo():
    """
    Demonstrates real-time camera feed processing.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nCamera Feed Demo")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display original and processed frames
        cv2.imshow('Webcam Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function to run all demonstrations.
    """
    print("OpenCV Demonstrations")
    print("====================")
    
    # 1. Basic Image Creation
    print("\n1. Basic Image Creation Demo")
    basic_image_creation()

    # 2. Color Blocks with Labels
    print("\n2. Color Blocks with Labels Demo")
    color_blocks_with_labels()

    # 3. Image Processing
    print("\n3. Image Processing Demo")
    image_path = 'photo.jpg'  # Replace with your image path
    processed_images = image_processing_demo(image_path)

    # 4. Camera Feed
    print("\n4. Camera Feed Demo")
    camera_feed_demo()

    print("\nAll demonstrations completed!")

if __name__ == "__main__":
    main()
