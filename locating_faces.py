import PIL.Image
import PIL.ImageDraw
import time
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

start_time = time.time()

image_path = "/Users/aadityabhatnagar/Downloads/College/College_Study_Material /CS F366 Facial Recog with Drones/database/students/2022A7PS0236U.png"
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    print("Error: Image could not be loaded. Please check the file path.")
    exit()

scale_percent = 150
width = int(image_bgr.shape[1] * scale_percent / 100)
height = int(image_bgr.shape[0] * scale_percent / 100)
resized_img = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])
sharpened_img = cv2.filter2D(resized_img, -1, kernel)

adjusted_img = cv2.convertScaleAbs(sharpened_img, alpha=1.2, beta=30)
image_rgb = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)

detector = MTCNN()
faces_mtcnn = detector.detect_faces(image_rgb)
confidence_threshold = 0.9
filtered_faces = [face for face in faces_mtcnn if face['confidence'] >= confidence_threshold]

number_of_faces = len(filtered_faces)
print("Found {} faces with confidence >= {}.".format(number_of_faces, confidence_threshold))

pil_image = PIL.Image.fromarray(image_rgb)
draw = PIL.ImageDraw.Draw(pil_image)

for face in filtered_faces:
    x, y, width, height = face['box']
    print(f"Location of face: Top: {y}, Left: {x}, Width: {width}, Height: {height}")
    draw.rectangle([x, y, x + width, y + height], outline="red")

pil_image.show()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")
