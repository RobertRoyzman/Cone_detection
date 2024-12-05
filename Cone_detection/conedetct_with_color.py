from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('best_150.pt')  # Replace with trained YOVOv8.pt file

# Input and output video paths
input_video_path = "fsd1.mp4"  # Input video
output_video_path = "fsd1_output.mp4"  # Output video

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving .mp4 files

# Define the VideoWriter to save the output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define a function to detect the dominant color in a region
def detect_dominant_color(roi):
    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for identification
    color_ranges = {
        "Orange": ((5, 100, 100), (15, 255, 255)),  # Hue range for orange
        "Yellow": ((20, 100, 100), (30, 255, 255)), # Hue range for yellow
        "Blue": ((100, 150, 0), (140, 255, 255)),   # Hue range for blue
        "Green": ((35, 50, 50), (85, 255, 255)), #Hue range for green
    }

    # Count pixels within each color range
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:
            return color_name

    return "Unknown"  # Default if no color is detected

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the current frame
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Process detections in the frame
    annotated_frame = frame.copy()  # Annotate the frame with YOLO's built-in method
    for box in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        confidence = box.conf.cpu().item()  # Confidence score
        class_id = int(box.cls.cpu().item())  # Class ID

        # Extract the region of interest (ROI) from the frame
        roi = frame[y1:y2, x1:x2]


        # Detect the dominant color in the ROI
        color = detect_dominant_color(roi)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),(0, 255, 0), 2)

        # Annotate the frame with the detected color
        label = f"Class: {class_id}, Color: {color}, Conf: {confidence:.2f}"
        text_y = max(y1 - 10,0)
        cv2.putText(annotated_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow("YOLOv8 Video Inference with Color Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_video_path}")