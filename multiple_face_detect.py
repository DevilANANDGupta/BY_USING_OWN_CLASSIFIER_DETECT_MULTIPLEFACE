import cv2
import sys
import tensorflow
import numpy as np

class Classifier:

    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=True)
        self.model = tensorflow.keras.models.load_model(self.model_path)

        
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        self.labels_path = labelsPath
        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = []
            for line in label_file:
                stripped_line = line.strip()
                self.list_labels.append(stripped_line)
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw= True, pos=(50, 50), scale=2, color = (0,255,0)):
        
        imgS = cv2.resize(img, (224, 224))

        image_array = np.asarray(imgS)
       
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

       
        self.data[0] = normalized_image_array

        
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]),
                        pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal

cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier("C:/Users/HP/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
maskClassifier = Classifier('D:/recognitionimage/converted_keras (2)/keras_model.h5','D:/recognitionimage/converted_keras (2)/labels.txt' )
model = models.load_model

video_capture = cv2.VideoCapture(0)


while True:
    
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    predection = maskClassifier.getPrediction(frame)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        # flags=cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame, 'OpenCV', 1, cv2.FONT_HERSHEY_COMPLEX ,1, (0, 255, 0), 0.5)

    # Display the resulting frame
    cv2.imshow('IDR Video Rendering', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()