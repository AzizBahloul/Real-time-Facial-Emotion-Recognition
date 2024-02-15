import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

# Load model
model = load_model("my_model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FacialEmotionRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_quit = tk.Button(window, text="Quit", width=10, command=self.quit)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

        self.start_time = time.time()  # Start time
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (64, 64))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels = np.expand_dims(img_pixels, axis=-1)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if time.time() - self.start_time < 30:
            self.window.after(10, self.update)
        else:
            self.quit()

    def quit(self):
        self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialEmotionRecognitionApp(root, "Facial Emotion Recognition")
