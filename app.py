# Import required modules
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import os
import cv2
import numpy as np
import pandas as pd
import random
import urllib.request
import re
import yt_dlp as youtube_dl
import pygame
import time
from keras.models import load_model
import base64

app = Flask(__name__)

# Load the emotion detection model
emotion_model = load_model('model.h5')

# Load the Haar Cascade for face detection
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize pygame mixer
pygame.init()
pygame.mixer.init()

# Emotion labels based on the model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        faceROI = gray[y:y+h, x:x+w]
        faceROI = cv2.resize(faceROI, (48, 48), interpolation=cv2.INTER_NEAREST)
        faceROI = np.expand_dims(faceROI, axis=0)
        faceROI = np.expand_dims(faceROI, axis=3)
        prediction = emotion_model.predict(faceROI)
        return emotion_labels[int(np.argmax(prediction))]

    return None

def sanitize_filename(filename):
    """Sanitize the filename by removing special characters."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c for c in filename if c in valid_chars)
    return sanitized.replace(" ", "_")

def song_recommendations(emotion):
    csv_name = f"Songs_name/{emotion}.csv"
    df = pd.read_csv(csv_name)
    data = df.values.tolist()
    length = len(data)
    selected_indices = random.sample(range(length), 10)
    song_names = [data[i][0].strip() for i in selected_indices]
    return song_names

def play_song_from_local_storage(song_name):
    output_path = os.path.join('Songs', song_name + '.mp3')
    if os.path.exists(output_path):
        try:
            pygame.mixer.music.load(output_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)
        except pygame.error:
            return None
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snapshot', methods=['POST'])
def snapshot():
    data = request.form['image_data']
    image_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    emotion = detect_emotion(img)
    if emotion:
        songs = song_recommendations(emotion)
        return jsonify({'emotion': emotion, 'songs': songs})
    else:
        return jsonify({'error': 'No face detected or emotion not recognized'})

@app.route('/download_play', methods=['POST'])
def download_play():
    song_name = request.form['song_name']
    result = play_song_from_local_storage(song_name)
    if result:
        return jsonify({'status': 'playing', 'song': song_name})
    else:
        return jsonify({'status': 'failed', 'error': 'Could not play the song'})

@app.route('/play', methods=['POST'])
def play_song():
    pygame.mixer.music.unpause()
    return jsonify({'status': 'playing'})

@app.route('/pause', methods=['POST'])
def pause_song():
    pygame.mixer.music.pause()
    return jsonify({'status': 'paused'})

if __name__ == '__main__':
    app.run(debug=True)