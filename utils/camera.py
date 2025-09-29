import cv2


class Camera:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)

    def get_frame(self):
        if not self.cap.isOpened():
            self.cap.open(self.stream_url)
        ret, frame = self.cap.read()
        if not ret:
            return None
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            return None
        return jpeg.tobytes()

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
