import sys
import argparse
import io
import datetime
from PIL import Image
import cv2
import torch
import numpy as np
from re import DEBUG, sub
import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from ultralytics import YOLO

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

@app.route("/")
def display_home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_image():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            print("Upload folder is ", filepath)
            f.save(filepath)
            global imgpath
            predict_image.imgpath = f.filename
            print("Printing predict_image :::::: ", predict_image)

            # Get file extension
            file_extension = f.filename.rsplit('.', 1)[1].lower()

            # Define the directory for saving predictions
            save_directory = os.path.join(basepath, 'runs', 'detect', 'predict')

            # Handle image files
            if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
                img = cv2.imread(filepath)
                frame = cv2.imencode(f'.{file_extension}', img)[1].tobytes()

                image = Image.open(io.BytesIO(frame))

                # Perform image detection
                yolo = YOLO(r"C:\Users\chris\Desktop\capstone project\Traffic_Vehicle_Real_Time_Detection\runs\detect\train\weights\best.pt")
                detections = yolo.predict(image, save=True, save_dir=save_directory)
                return redirect(url_for('display', filename=f.filename))
            
            # Handle video files
            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                # Get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

                # Initialize YOLO model
                model = YOLO(r"C:\Users\chris\Desktop\capstone project\Traffic_Vehicle_Real_Time_Detection\runs\detect\train\weights\best.pt")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detect objects in each frame with YOLO
                    results = model(frame, save=True)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("results", res_plotted)

                    # Write the frame to the output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break

                return video_feed()

    return render_template("index.html")

#This is the display function that is used to serve the image or video from the folder_path directory
@app.route('/<path:filename>')
def display(filename):
    # Check if the YOLO predictions are saved in the correct directory
    folder_path = os.path.join('runs', 'detect', 'predict')
    
    # Ensure the folder exists and has files
    if not os.path.exists(folder_path):
        return "Prediction folder does not exist.", 404

    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not subfolders:
        return "No predictions available.", 404
    
    # Get the latest prediction folder
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = os.path.join(folder_path, latest_subfolder)
    print("Printing directory: ", directory)
    
    # Check if there are any files in the folder
    files = os.listdir(directory)
    if not files:
        return "No files found in the directory.", 404
    
    latest_file = files[0]
    print("Latest file: ", latest_file)
    
    # Serve the latest file
    file_extension = latest_file.rsplit('.', 1)[1].lower()
    
    if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
        return send_from_directory(directory, latest_file)
    else:
        return "Invalid file format"
    
def get_frame():
    folder_path = os.getcwd()
    mp4_files = "output.mp4"
    video = cv2.VideoCapture(mp4_files)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

#function to display the detected objects on video on html page
@app.route("/video_feed")
def video_feed():
    print("function called")
    return Response(get_frame(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


