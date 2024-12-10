import cv2
import os

# Video file path
video_path = "IMG_4294.mov"
output_folder = "extracted_frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Extract frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Format frame filename
    frame_filename = f"frame_{frame_count:06d}.jpg"
    frame_path = os.path.join(output_folder, frame_filename)
    
    # Save frame as image
    cv2.imwrite(frame_path, frame)
    print(f"Saved: {frame_filename}")
    
    frame_count += 1

# Release resources
cap.release()
print("Frame extraction completed.")
