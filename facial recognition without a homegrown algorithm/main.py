import face_recognition
import os
import cv2
import numpy as np

def calculate_face_confidence(face_distance, face_match_threshold=0.6):
    """
    Calculate the confidence of a face match based on the face distance and a threshold.
    """
    face_range = (1.0 - face_match_threshold)
    linear_value = (1.0 - face_distance) / (face_range * 2.0)

    if face_distance > face_match_threshold:
        return f'{round(linear_value * 100, 2)}%'
    else:
        value = (linear_value + ((1.0 - linear_value) * ((linear_value - 0.5) * 2) ** 0.2)) * 100
        return f'{round(value, 2)}%'

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """
        Load known faces from the 'faces' directory and encode them.
        """
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Source not found')

        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = 'Unknown'
                confidence = 'Unknown'

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = calculate_face_confidence(face_distances[best_match_index])

                self.face_names.append(f'{name} ({confidence})')

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('e'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
