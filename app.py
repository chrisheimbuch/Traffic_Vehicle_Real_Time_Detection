from flask import Flask, render_template, request, redirect, url_for, Response
import cv2  # OpenCV to process images
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load your YOLO model
yolo_model = YOLO(r"C:\Users\chris\Desktop\capstone project\Traffic_Vehicle_Real_Time_Detection\runs\detect\train\weights\best.pt")

# List of class names corresponding to your YOLO model (customize this to match your model)
class_names = ['bus', 'car', 'motorbike', 'threewheel', 'truck', 'van']  # Update to match your model

# Function to process frames and return them with YOLO detection results
def gen_frames():
    # Open the webcam (use 0 for the default webcam or change it if you have multiple cameras)
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame with YOLO model
        results = yolo_model.predict(source=frame, save=False)
        
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            labels = result.boxes.cls  # Class labels (indices)
            confidences = result.boxes.conf  # Confidence scores
            
            for box, label, confidence in zip(boxes, labels, confidences):
                # Get the class name based on the label index
                class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class {int(label)}"
                
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box with thicker lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw class name and confidence score
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Encode the frame in JPEG format and yield the result
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frame bytes

    cap.release()

@app.route('/webcam')
def webcam_feed():
    """Route to start the webcam feed and display it."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_detection', methods=['POST'])
def video_detection():
    file = request.files['video']  # Get the uploaded video
    video_path = 'static/uploaded_video.mp4'
    
    # Save the uploaded video to disk
    file.save(video_path)

    # Open the video with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame with YOLO model
        results = yolo_model.predict(source=frame, save=False)
        
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            labels = result.boxes.cls  # Class labels (indices)
            confidences = result.boxes.conf  # Confidence scores
            
            for box, label, confidence in zip(boxes, labels, confidences):
                class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class {int(label)}"
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save or process the frames here (e.g., for displaying results)
        
    cap.release()

    # Return the video feed or redirect to a results page
    return render_template('index.html', video_path=video_path)  # You can adjust this as per your layout

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        file = request.files['image']  # Get the uploaded image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Process the image with YOLO model
        results = yolo_model.predict(source=image, save=False)
        
        # Initialize a list to store class names and confidence scores
        classifications = []
        
        # Draw the bounding boxes and labels on the image
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            labels = result.boxes.cls  # Class labels (indices)
            confidences = result.boxes.conf  # Confidence scores
            
            for box, label, confidence in zip(boxes, labels, confidences):
                # Get the class name based on the label index
                class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class {int(label)}"
                
                # Append both the class name and confidence score to the list
                classifications.append({
                    'class': class_name.title(),
                    'confidence': round(float(confidence) * 100)  # Convert to percentage and round to whole number
                })
                
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box with thicker lines (thickness = 3)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
                
                # Draw larger class name and confidence score (font scale = 1.2, thickness = 3)
                cv2.putText(image, f"{class_name} ({confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)  # Set text color to red and larger font
        
        # Resize the image to make it larger (e.g., 1.5x the original size)
        image = cv2.resize(image, (int(image.shape[1] * 1.5), int(image.shape[0] * 1.5)))
        
        # Save the processed image
        processed_image_path = 'static/processed_image.jpg'
        cv2.imwrite(processed_image_path, image)
        
        # Return the page with the processed image and classification details
        return render_template('index.html', image_path=processed_image_path, classifications=classifications)

    # Default GET request just renders the page for upload
    return render_template('index.html')

# Add a route to handle re-uploading
@app.route('/reupload')
def reupload():
    return redirect(url_for('upload_and_process'))

if __name__ == '__main__':
    app.run(debug=True)