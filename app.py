from flask import Flask, render_template, request
import cv2  # OpenCV to process images
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load your YOLO model
yolo_model = YOLO(r"C:\Users\chris\Desktop\capstone project\Traffic_Vehicle_Real_Time_Detection\runs\detect\train\weights\best.pt")

# List of class names corresponding to your YOLO model (customize this to match your model)
class_names = ['bus', 'car', 'motorbike', 'threewheel', 'truck', 'van']  # Update to match your model

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        file = request.files['image']  # Get the uploaded image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Process the image with YOLO model
        results = yolo_model.predict(source=image, save=False)  # Run YOLO detection
        
        # Initialize variables to store predictions and accuracy
        accuracies = []
        
        # Draw the bounding boxes and labels on the image
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            labels = result.boxes.cls  # Class labels (indices)
            confidences = result.boxes.conf  # Confidence scores
            
            for box, label, confidence in zip(boxes, labels, confidences):
                accuracies.append(round(float(confidence) * 100))  # Convert to percentage and round to whole number
                x1, y1, x2, y2 = map(int, box)
                
                # Get the class name based on the label index
                class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class {int(label)}"
                
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
        
        # Return the page with the processed image and accuracy percentage
        return render_template('index.html', image_path=processed_image_path, accuracies=accuracies)

    # Default GET request just renders the page for upload
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)