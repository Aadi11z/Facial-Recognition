import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import face_recognition
import logging
import Facial_Recognition.X as X

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebcamFaceRecognizer:
    def __init__(self, confidence_threshold=0.9, distance_threshold=0.5, camera_index=0):
        self.width = 1500
        self.height = 1500
        self.camera_index = camera_index
        
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
        
    def process_face(self, face_encoding):
        if face_encoding is None:
            return "Unknown", 1.0
        distances = face_recognition.face_distance(self.known_faces, face_encoding)
        min_distance = min(distances) if distances.size else 1.0
        if min_distance <= self.distance_threshold:
            match_index = np.argmin(distances)
            return self.known_names[match_index], min_distance
        return "Unknown", min_distance

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return
        logger.info("Starting webcam face recognition...")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enhanced_frame = self.enhance_image(frame_rgb)
                faces = self.detector.detect_faces(enhanced_frame)
                faces = [face for face in faces if face['confidence'] >= self.confidence_threshold]
                for face in faces:
                    x, y, w, h = face['box']
                    face_location = (y, x + w, y + h, x)
                    face_encodings = face_recognition.face_encodings(enhanced_frame, [face_location])
                    if face_encodings:
                        name, confidence = self.process_face(face_encodings[0])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{name} ({confidence:.2f})"
                        cv2.putText(frame, label, (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Webcam Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam face recognition stopped")

if __name__ == "__main__":
    recognizer = WebcamFaceRecognizer(camera_index=0)
    recognizer.run()
