**Cone detection**

# how to run
use this prompt in terminal panel
```pip install ultralytics==8.2.103 ```

in the main program replace the path to the downloaded YOLO model (.pt)
and the paths to the preferred  input video and output video upload location.

# Load the YOLO model
model = YOLO('.pt_file_here')  # Replace with trained YOVOv8.pt file

# Input and output video paths
input_video_path = "input_video_here"  # Input video
output_video_path = "output_video_here"  # Output video

recommended input file can be found in the BGRacing Autonomous team home assignment.

