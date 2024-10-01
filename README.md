﻿# Vehicle Real Time Detection - YOLO v8 Capstone Project

- **[Documents](./documents):** includes PDF documents of slides used in class presentation and of the application/website

- **[Source](./source):** includes source code for the entire project

This is a Phase 5 Capstone Project for Flatiron School's Datascience Bootcamp. I created a real time object detection application where a user can upload an image, video, or opt to use real time detection via a webcam of their car, truck, bus, motorcycle, van or tricycle or a picture of the class to have the model detect it in a given photo or video. I trained a YOLO v8 nano model on a traffic image dataset on 6 different classes to learn complex heuristics between each class. The frontend is written in HTML and styled using CSS and the backend is written in python using Flask. 

### How to Use

<img width="1262" alt="websiteHomePage" src="https://github.com/user-attachments/assets/35abb5d6-6078-4958-a138-4b16317dcb59">

Once you are at the homepage of the website, users have the option to either select to upload an image of a vehicle, upload a video including vehicles, or use your webcam for real time detection. Select the image you want to use and click on upload image and detect.

### Images

![image](https://github.com/user-attachments/assets/682670c1-7490-4174-ab41-e833f9df5a6c)

For images, in the example above, once the object detection has been performed, it will include a bounding box and classification as well as a percentage of accuracy the model is sure of what type of class it is. Additionally, the classes and accuracy percentages will be displayed at the bottom of the web page below the image. The user will have the option to either download the image that was identified with a particular vehicle class or returning to the home page to test another image or video. 

### Videos

![image](https://github.com/user-attachments/assets/3ff7a7d8-ccce-478c-b8b0-a9dbec20f85e)

For videos, in the example above, once the object detection has been performed, it will include bounding boxes and classifications and accuracy percentage scored for whatever class is detected in the video. The user will have the option to download the video that was processed if desired, or return to the home page to test another image or video.

### Real Time Detection

For real time detection, make sure you have a webcam configured. All the user needs to do is click on the last button on the home page and make sure the camera is pointed at a vehicle or vehicles.
