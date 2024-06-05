import time

import cv2
import numpy as np
import threading
import mediapipe as mp
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from grammerbn import grammer

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load LSTM model and actions
modelg = tf.keras.models.load_model("Pleasework.h5")
actions = np.load('actionsnew.npy')


# Load labels
with open('LabelList.txt', 'r') as file:
    labels = [line.strip() for line in file.readlines()]

# Define font properties
fontpath = "./Li Sirajee Sanjar Unicode.ttf"
font = ImageFont.truetype(fontpath, 32)
text_color = (255, 255, 255, 255)

# Define the function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Define the function to extract keypoints from the MediaPipe results
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


# Define the function to capture frames from the webcam
def capture_frames(queue):
    global display_text
    global key
    global key_flag
    global start_queuing
    global classification_flag
    global display_text_lock
    global should_continue

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    num_frames_captured = 0
    while should_continue.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # queue.put(frame)
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), display_text, font=font, fill=text_color)

        frame = np.array(img_pil)

        xmin = 163
        ymin = 101
        xmax = 498
        ymax = 400

        # Draw the bounding box
        bodybb = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        face_xmin = 270
        face_xmax = 370
        face_ymin = 60
        face_ymax = 180

        facebb = cv2.rectangle(frame, (face_xmin, face_ymin), (face_xmax, face_ymax), (0, 0, 255), 2)
        cv2.imshow('Webcam Stream', facebb)

        key = cv2.waitKey(1)
        if key != -1:
            key_flag.set()

        if start_queuing.is_set():
            queue.put(frame)
            num_frames_captured += 1

        if num_frames_captured == 60:
            display_text = "Processing..."
            start_queuing.clear()
            num_frames_captured = 0

        if key == ord('q'):
            should_continue.clear()
            frame_queue.queue.clear()
            landmark_queue.queue.clear()
            while not frame_queue.empty():
                frame_queue.get()
            while not landmark_queue.empty():
                landmark_queue.get()
            break

    cap.release()
    cv2.destroyAllWindows()


# Define the function to extract landmarks
def extract_landmarks(queue_in, queue_out, holistic_model):
    global capture_flag
    global classification_flag
    global key
    global display_text
    global display_text_lock

    while should_continue.is_set():
        if key == ord('q'):
            break
        capture_flag.wait()
        start_time = time.time()
        print("reached extract_landmarks")
        frames_with_landmarks = []

        for _ in range(60):  # Capture 60 frames sequentially
            display_text = "Capturing for landmarks..."
            #if queue_in.empty():
            #    time.sleep(1 / 60)
            frame = queue_in.get()
            image, results = mediapipe_detection(frame, holistic_model)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                frames_with_landmarks.append(keypoints)

        print("Length of frames with landmarks:", len(frames_with_landmarks))
        # If there are more than 30 frames with landmarks, resample to get 30 evenly spaced frames
        if len(frames_with_landmarks) > 30:
            display_text = "Refactoring landmarks..."
            indices = np.round(np.linspace(0, len(frames_with_landmarks) - 1, 30)).astype(int)
            frames_with_landmarks = [frames_with_landmarks[i] for i in indices]

        # Send the keypoints of 30 frames with landmarks for classification
        for keypoints in frames_with_landmarks:
            display_text = "Refactoring landmarks..."
            queue_out.put(keypoints)

        print("Length of queue out", queue_out.qsize())

        end_time = time.time()
        landmark_extraction_time = end_time - start_time
        print("Time spent on landmark extraction:", landmark_extraction_time, "seconds")
        capture_flag.clear()
        classification_flag.set()


# Define the function to classify words
def classify_landmarks(queue):
    global modelg
    global classification_flag
    global predicted_words
    global key
    global display_text
    global display_text_lock

    while should_continue.is_set():
        if key == ord('q'):
            break
        classification_flag.wait()

        display_text = "Classifying landmarks..."
        # Start time measurement for classification
        start_time = time.time()

        sequence = []
        print("reached classify_landmarks")
        print(queue.qsize())
        while not queue.empty():
            keypoints = queue.get()
            sequence.append(keypoints)
            print(len(sequence))

        if len(sequence) < 30:  # Check if the sequence is less than 30
            if sequence:  # Ensure there is at least one item to repeat
                last_keypoint = sequence[-1]
                sequence.extend([last_keypoint] * (30 - len(sequence)))  # Fill the sequence
                print("Filled up rest of the landmarks")

        if len(sequence) == 30:  # When 30 frames have been processed
            display_text = "Classifying landmarks..."
            sequence_array = np.array(sequence)
            prediction = modelg.predict(np.expand_dims(sequence_array, axis=0))[0]
            predicted_label = labels[np.argmax(prediction)]
            predicted_words.append(predicted_label)
            display_text = "Done"
            print(predicted_label)

        # End time measurement for classification
        end_time = time.time()
        classification_time = end_time - start_time
        print("Time spent on classification:", classification_time, "seconds")

        classification_flag.clear()


# Initialize queues and shared variables
frame_queue = Queue(maxsize=60)
landmark_queue = Queue(maxsize=60)
display_text = "Press 's' to get input. Press 'd' for output"
capture_flag = threading.Event()
classification_flag = threading.Event()
predicted_words = []
key = -1
key_flag = threading.Event()
start_queuing = threading.Event()
display_text_lock = threading.Lock()

should_continue = threading.Event()
should_continue.set()

# Create and start the frame capture process
thrd_frame_capture = threading.Thread(target=capture_frames, args=(frame_queue,))
thrd_frame_capture.start()

# Create and start the landmark extraction process
thrd_extract_landmarks = threading.Thread(target=extract_landmarks, args=(frame_queue,
                                                                          landmark_queue, holistic))
thrd_extract_landmarks.start()

# Create and start the classification process
thrd_classify_landmarks = threading.Thread(target=classify_landmarks, args=(landmark_queue,))
thrd_classify_landmarks.start()

# Main loop
while True:
    key_flag.wait()
    if key == ord('s'):  # S is pressed
        display_text = "Pressed"

        for i in range(3, 0, -1):
            display_text = f"Starting in {i}..."
            time.sleep(1)  # Wait for 1 second between counts

        start_queuing.set()
        capture_flag.set()
        display_text = "Starting capture"
        key_flag.clear()

    if key == ord('1'):  # 1 is pressed
        modelg = tf.keras.models.load_model("Pleaseworknew.h5")
        
        with open('label2.txt', 'r') as file:
            labels = [line.strip() for line in file.readlines()]

        # print("Model 1 loaded")
        key_flag.clear()

    if key == ord('2'):  # 2 is pressed
        modelg = tf.keras.models.load_model("Pleasework.h5")

        with open('LabelList.txt', 'r') as file:
            labels = [line.strip() for line in file.readlines()]

          #  print("Model 2 loaded")
            key_flag.clear()

    elif key == ord('d'):
        start_time = time.time()
        display_text = "Displaying sentence..."
        print(predicted_words)
        incorrect_sentence = ' '.join(predicted_words)
        # print(incorrect_sentence)
        corrected_sentences = grammer(incorrect_sentence)
        # print(corrected_sentences)
        display_text = f"Sentence: {corrected_sentences}"
        end_time = time.time()
        display_time = end_time - start_time
        print("Time spent on sentence form:", display_time, "seconds")

        predicted_words = []
        key_flag.clear()
    elif key == ord('q'):  # Q is pressed
        should_continue.clear()
        frame_queue.queue.clear()
        landmark_queue.queue.clear()
        while not frame_queue.empty():
            frame_queue.get()
        while not landmark_queue.empty():
            landmark_queue.get()

        thrd_frame_capture.join()
        thrd_extract_landmarks.join()
        thrd_classify_landmarks.join()

        key_flag.clear()

        break

# Join threads and release resources
thrd_frame_capture.join()
thrd_extract_landmarks.join()
thrd_classify_landmarks.join()
holistic.close()