# import cv2
# import numpy as np
# def ghibli_filter(image_path, output_path="ghibli_output.jpg"):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (800, 800))  # Resize for consistency
#     # Step 1: Apply bilateral filter multiple times
#     smooth = img.copy()
#     for _ in range(5):
#         smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)
#     # Step 2: Edge detection for anime outlines
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.medianBlur(gray, 5)
#     edges = cv2.adaptiveThreshold(edges, 255,
#                                    cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY, 9, 9)
#     # Step 3: Color quantization (using k-means)
#     Z = img.reshape((-1, 3))
#     Z = np.float32(Z)
#     K = 8  # You can tweak this for different styles
#     _, label, center = cv2.kmeans(Z, K, None,
#                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
#                                   10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)
#     quantized = center[label.flatten()]
#     quantized = quantized.reshape((img.shape))
#     # Step 4: Combine quantized image with edges
#     cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
#     # Save and return
#     cv2.imwrite(output_path, cartoon)
#     return cartoon

# image = ghibli_filter('photo.jpg')
# while True:
#     cv2.imshow('Ghibli Filter', image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# def cartoonize_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (800, 600))

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, 5)

#     # Detect edges
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                   cv2.THRESH_BINARY, 9, 10)

#     # Apply bilateral filter to smooth colors
#     color = cv2.bilateralFilter(img, 9, 250, 250)

#     # Combine edges and color
#     cartoon = cv2.bitwise_and(color, color, mask=edges)

#     cv2.imshow("Cartoonized Image", cartoon)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# cartoonize_image("photo.jpg")



import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load the input image
input_image_path = "nitya.jpg"
input_image = cv2.imread(input_image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(input_image, (256, 256))  # Resize for faster processing
input_image = np.array(input_image) / 255.0  # Normalize to [0, 1]



# Load a Ghibli-style reference image
style_image_path = "ghibli_style.jpg"
style_image = cv2.imread(style_image_path)
style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
style_image = cv2.resize(style_image, (256, 256))
style_image = np.array(style_image) / 255.0

# Apply style transfer
input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
style_tensor = tf.convert_to_tensor(style_image, dtype=tf.float32)
input_tensor = tf.expand_dims(input_tensor, axis=0)
style_tensor = tf.expand_dims(style_tensor, axis=0)

stylized_image = model(input_tensor, style_tensor)[0]
stylized_image = tf.squeeze(stylized_image).numpy()

# Save the output
output_image_path = "ghibli_art.jpg"
stylized_image = (stylized_image * 255).astype(np.uint8)
stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_image_path, stylized_image)

# Display the output
cv2.imshow("Ghibli Art", stylized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()