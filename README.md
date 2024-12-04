# ParkingSpotCounter
This project is a real-time parking lot occupancy detection system that uses machine learning to identify whether parking spots are empty or occupied in a video feed. The system is designed to process video input of a parking lot, analyze individual parking spots, and overlay visual indicators (rectangles and counters) to show the occupancy status.


# Features
- Machine Learning-Based Detection:
A pre-trained machine learning model is used to classify parking spots as empty or occupied based on image data.
- Connected Component Analysis:
A parking lot mask is processed using OpenCV's connected components to detect and define parking spots.
- Real-Time Video Processing:
Processes video frames at intervals to optimize performance.
Overlays green rectangles for empty spots and red rectangles for occupied spots on the video feed.
<img width="954" alt="Screenshot 2024-12-04 at 15 35 51" src="https://github.com/user-attachments/assets/daaa4031-3bc8-407f-8e91-032b4b0afb71">

# Necessary Libraries 
 - OpenCV
 - Scikit-Image
 - Numpy
 - Pickle


