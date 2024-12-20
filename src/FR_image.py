import os
import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import logging
import Facial_Recognition.X as X

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, confidence_threshold=0.9, distance_threshold=0.5):
        self.known_faces = [
            X.jag_encoding, X.std0136_encoding, X.std0164_encoding,
            X.std0172_encoding, X.std0174_encoding, X.std0175_encoding,
            X.std0177_encoding, X.std0179_encoding, X.std0180_encoding,
            X.std0184_encoding, X.std0185_encoding, X.std0186_encoding,
            X.std0187_encoding, X.std0188_encoding, X.std0191_encoding,
            X.std0193_encoding, X.std0194_encoding, X.std0195_encoding,
            X.std0196_encoding, X.std0199_encoding, X.std0200_encoding,
            X.std0201_encoding, X.std0202_encoding, X.std0203_encoding,
            X.std0207_encoding, X.std0210_encoding, X.std0211_encoding,
            X.std0212_encoding, X.std0214_encoding, X.std0216_encoding,
            X.std0218_encoding, X.std0219_encoding, X.std0220_encoding,
            X.std0224_encoding, X.std0225_encoding, X.std0226_encoding,
            X.std0229_encoding, X.std0230_encoding, X.std0231_encoding,
            X.std0234_encoding, X.std0235_encoding, X.std0237_encoding, 
            X.std0239_encoding, X.std0240_encoding, X.std0241_encoding, 
            X.std0242_encoding, X.std0243_encoding, X.std0245_encoding, 
            X.std0246_encoding, X.std0247_encoding, X.std0249_encoding, 
            X.std0250_encoding, X.std0903_encoding, X.std0274_encoding
 
        ]

        self.known_names = [
            "Jagadish Sir", "2022A7PS0136U", "2022A7PS0164U",
            "2022A7PS0172U", "2022A7PS0174U", "2022A7PS0175U",
            "2022A7PS0177U", "2022A7PS0179U", "2022A7PS0180U",
            "2022A7PS0184U", "2022A7PS0185U", "2022A7PS0186U",
            "2022A7PS0187U", "2022A7PS0188U", "2022A7PS0191U",
            "2022A7PS0193U", "2022A7PS0194U", "2022A7PS0195U",
            "2022A7PS0196U", "2022A7PS0199U", "2022A7PS0200U",
            "2022A7PS0201U", "2022A7PS0202U", "2022A7PS0203U",
            "2022A7PS0207U", "2022A7PS0210U", "2022A7PS0211U",
            "2022A7PS0212U", "2022A7PS0214U", "2022A7PS0216U",
            "2022A7PS0218U", "2022A7PS0219U", "2022A7PS0220U",
            "2022A7PS0224U", "2022A7PS0225U", "2022A7PS0226U",
            "2022A7PS0229U", "2022A7PS0230U", "2022A7PS0231U",
            "2022A7PS0234U", "2022A7PS0235U", "2022A7PS0237U", 
            "2022A7PS0239U", "2022A7PS0240U", "2022A7PS0241U", 
            "2022A7PS0242U", "2022A7PS0243U", "2022A7PS0245U", 
            "2022A7PS0246U", "2022A7PS0247U", "2022A7PS0249U", 
            "2022A7PS0250U", "2022A7PS0903U", "2022A7PS0274U"
        ]
        
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.detector = MTCNN() 

    def enhance_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return cv2.fastNlMeansDenoisingColored(enhanced)

    def preprocess_image(self, image_path):
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise ValueError("Image failed")
            
            resized = cv2.resize(image_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            enhanced = self.enhance_image(image_rgb)

            faces = self.detector.detect_faces(enhanced)
            filtered_faces = [face for face in faces if face['confidence'] >= self.confidence_threshold]
            logger.info(f"Found {len(filtered_faces)} faces with confidence >= {self.confidence_threshold}")
            
            return enhanced, filtered_faces

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def process_face(self, face_encoding):
        if face_encoding is None:
            return "Unknown", 1.0
        
        distances = face_recognition.face_distance(self.known_faces, face_encoding)
        min_distance = min(distances) if distances.size else 1.0
        
        if min_distance <= self.distance_threshold:
            match_index = np.argmin(distances)
            return self.known_names[match_index], min_distance
        return "Unknown", min_distance

    def recognize(self, image_path):
        try:
            image, detected_faces = self.preprocess_image(image_path)
            face_locations = [(face['box'][1], face['box'][0] + face['box'][2],
                               face['box'][1] + face['box'][3], face['box'][0]) 
                              for face in detected_faces]
            
            face_encodings = face_recognition.face_encodings(image, face_locations)
            results = []
            
            for encoding, location in zip(face_encodings, face_locations):
                name, confidence = self.process_face(encoding)
                results.append((name, confidence, location))

            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            for name, confidence, (top, right, bottom, left) in results:
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
                label = f"{name} ({confidence:.2f})"
                draw.text((left + 6, bottom + 5), label, fill=(255, 255, 255))

            logger.info(f"Recognition completed in {time.time() - start_time:.2f} seconds")
            return pil_image

        except Exception as e:
            logger.error(f"Recognition failed: {str(e)}")
            raise

if __name__ == "__main__":
    start_time = time.time()
    recognizer = FaceRecognizer()
    result = recognizer.recognize(
        "/Users/aadityabhatnagar/Downloads/College/College_Study_Material /CS F366 Facial Recog with Drones/database/test/CameraPhoto20240207-125550.png"
        )
    result.show()
