import cv2
import time
import numpy as np
import tkinter as tk
import threading
import matplotlib.pyplot as plt
from HeartRateMonitor import HeartRateMonitor
from MultiFrame import MultiFrame
from DataManager import DataManager


class VideoLab:
    def __init__(self, camera, buffer_size=50, min_face_size=(32, 32), scale_factor=1.1, min_neighbors=6,
                 forehead_tracking=True):
        """
        Initialize the VideoLab
        """
        self.video_feed = cv2.VideoCapture(camera) # Feed for video date (Change to video file if needed)
        self.buffer_size = buffer_size # Size of captured frames for the computation
        self.min_face_size = min_face_size # Minimum size of faces to count as such
        self.scale_factor = scale_factor # Scale factor influencing face detection (adjust for difficult captures)
        self.min_neighbors = min_neighbors # Neighbor factor influencing face detection(adjust for difficult captures)
        self.forehead_tracking = forehead_tracking # Control variable for a found forehead
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # HaarCascade for faces
        self.heart_rate_monitor = HeartRateMonitor() # Object for calculating a heart rate
        self.heart_data = None # Buffer of heart rates and time values
        self.heart_rate = 0 # Current heart rate (BPM)
        self.current_frame = None
        self.prev_frame = None
        self.current_face = None
        self.tracker = None  # Face tracker object for smooth bounding boxes to not detect new faces for every frame
        self.tracking_face = False # Control variable for tracking instead of capturings faces

    def start_recording(self):
        """
        Start the video feed and process frames
        """
        self.video_feed = cv2.VideoCapture(0)
        ret, frame = self.video_feed.read()
        self.prev_frame = MultiFrame(frame)

        while True:
            ret, frame = self.video_feed.read()
            if not ret:
                print('Failed to retrieve frame.')
                break

            self.current_frame = MultiFrame(frame)

            if not self.tracking_face:
                self.process_faces()

            if self.tracking_face:
                self.process_tracked_face()

            self.prev_frame = self.current_frame  # Update the prev_frame

            self.show_frames()

            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == ord('q'):
                break

        self.video_feed.release()
        cv2.destroyAllWindows()

    def process_faces(self):
        """
        Detect faces in the frame and process them
        """
        faces = self.face_cascade.detectMultiScale(self.current_frame.rgb, scaleFactor=self.scale_factor,
                                                   minNeighbors=self.min_neighbors, minSize=self.min_face_size,
                                                   flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            self.current_face = self.get_best_face(faces)
            (x, y, w, h) = self.current_face
        else:
            self.current_face = None

        # Draw a rectangle around the face
        if self.current_face is not None:
            (x, y, w, h) = self.current_face
            
            cv2.rectangle(self.current_frame.rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display heart rate over the face
            if self.heart_rate is not None:
                heart_rate_text = 'Heart Rate: ' + str(self.heart_rate) + ' BPM'
                cv2.putText(self.current_frame.rgb, heart_rate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def process_tracked_face(self):
        """
        Update the tracker and process the tracked face
        """
        tracking_ok, bbox = self.tracker.update(self.current_frame.rgb)

        if tracking_ok:
            (x, y, w, h) = [int(v) for v in bbox]
            self.current_face = (x, y, w, h)

            if self.forehead_tracking:
                self.process_forehead()

            # Draw a rectangle around the face
            cv2.rectangle(self.current_frame.rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display heart rate over the face
            if self.heart_rate is not None:
                heart_rate_text = 'Heart Rate: ' + str(self.heart_rate) + ' BPM'
                cv2.putText(self.current_frame.rgb, heart_rate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def get_best_face(self, face_list):
        """
        Get the best face from the list of detected faces
        """
        if self.current_face is None:
            return max(face_list, key=lambda f: f[1])
        else:
            return min(face_list, key=lambda f: self.bb_distance(f, self.current_face))

    def start_tracking_face(self):
        """
        Start tracking the face using the KCF tracker
        """
        if self.current_face is not None:
            (x, y, w, h) = self.current_face
            self.tracker = cv2.TrackerKCF_create()  # Create a KCF tracker object
            self.tracker.init(self.current_frame.rgb, (x, y, w, h))  # Initialize the tracker with the face bounding box
            self.tracking_face = True

    def stop_tracking_face(self):
        """
        Stop tracking the face
        """
        self.tracking_face = False

    def process_forehead(self):
        """
        Process the forehead region to calculate heart rate
        """
        (x, y, w, h) = self.current_face
        (fx, fy, fw, fh) = self.get_forehead((x, y, w, h))
        (fx, fy, fw, fh) = (int(fx), int(fy), int(fw), int(fh))

        forehead = self.current_frame.rgb[fy:fy + fh, fx:fx + fw]
        self.heart_rate = self.heart_rate_monitor.compute_heart_rate(forehead)

        if self.heart_rate is not None and self.is_valid_heart_rate(self.heart_rate) and self.heart_data is not None:
            self.heart_data.append((time.time(), self.heart_rate))
            cv2.circle(self.current_frame.rgb, (20, 20), 10, (255, 0, 0), -1)

    def is_valid_heart_rate(self, heart_rate):
        """
        Check if the heart rate is valid (within a reasonable range)
        """
        return 40 < heart_rate < 200

    def show_frames(self):
        """
        Show the frames with face detection and heart rate information
        """
        if self.current_face is not None:
            (x, y, w, h) = self.current_face
            cv2.rectangle(self.current_frame.rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("RGB Image", self.current_frame.rgb)

    def stop_recording(self):
        """
        Release the video feed and close the OpenCV windows
        """
        self.video_feed.release()
        cv2.destroyAllWindows()

    def start_measurement_thread(self):
        """
        Start a thread to measure heart rate in real-time
        """
        self.heart_data = []
        self.create_sliders()
        measurement_thread = threading.Thread(target=self.measure_heart_rate)
        measurement_thread.start()

    def measure_heart_rate(self):
        """
        Measure heart rate by recording frames and processing the forehead region
        """
        self.heart_data = []
        self.start_recording()

    def save_and_exit(self):
        """
        Stop recording, save the captured data to a CSV file, and exit
        """
        self.stop_recording()
        if len(self.heart_data) > 0:
            dm = DataManager('data.csv', self.heart_data)
            dm.save_to_csv()
            print('Saved captured data to data.csv')

    def show_realtime_plot(self):
        """
        Show a real-time plot of heart rate data using matplotlib
        """
        plt.ion()
        plt.figure()
        plt.title('Heart Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Heart Rate (BPM)')
        x_data, y_data = [], []
        line, = plt.plot(x_data, y_data)
        while True:
            x_data, y_data = zip(*self.heart_data)
            line.set_xdata(x_data)
            line.set_ydata(y_data)
            if len(x_data) > 0:
                plt.xlim(max(0, x_data[-1] - 10), x_data[-1])

            # Set the y-axis limits to show heart rate values in the range of 40 to 200 BPM
            plt.ylim(40, 200)
            plt.draw()
            plt.pause(0.1)

    def get_forehead(self, face):
        """
        Simple calculation to estimate the head's forehead
        """
        (x, y, w, h) = face
        x += w * 0.5
        y += h * 0.13
        w *= 0.25
        h *= 0.15

        x -= (w / 2.0)
        y -= (h / 2.0)

        return (x, y, w, h)

    def bb_distance(self, a, b):
        """
        Distance measure of two bounding boxes
        """
        (ax, ay, _, _) = a
        (bx, by, _, _) = b
        return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


    def create_sliders(self):
        # Create a slider for scale_factor
        self.scale_factor_label = tk.Label(root, text="Scale Factor")
        self.scale_factor_label.pack(pady=5)
        self.scale_factor_slider = tk.Scale(root, from_=1.1, to=1.9, resolution=0.1, orient=tk.HORIZONTAL,
                                            command=self.update_scale_factor)
        self.scale_factor_slider.set(self.scale_factor)
        self.scale_factor_slider.pack()

        # Create a slider for min_neighbors
        self.min_neighbors_label = tk.Label(root, text="Min Neighbors")
        self.min_neighbors_label.pack(pady=5)
        self.min_neighbors_slider = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL,
                                             command=self.update_min_neighbors)
        self.min_neighbors_slider.set(self.min_neighbors)
        self.min_neighbors_slider.pack()

        root.update()

    def update_scale_factor(self, value):
        self.scale_factor = float(value)

    def update_min_neighbors(self, value):
        self.min_neighbors = int(value)

# Main script
if __name__ == "__main__":
    vl = VideoLab(0)

    root = tk.Tk()
    root.title("Heart Rate Monitor")
    root.geometry("275x350")

    start_button = tk.Button(root, text="Initiate Video Feed", command=vl.start_measurement_thread)
    start_button.pack(pady=5)

    tracking_button = tk.Button(root, text="Start Face Recording", command=vl.start_tracking_face)
    tracking_button.pack(pady=5)

    plot_button = tk.Button(root, text="Show Real-Time Plot", command=vl.show_realtime_plot)
    plot_button.pack(pady=10)

    stop_button = tk.Button(root, text="Stop Recording & Save Data", command=vl.save_and_exit)
    stop_button.pack(pady=20)

    root.mainloop()
