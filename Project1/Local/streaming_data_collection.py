import cv2
import os

dropped = 0  # Drop frames count
frame_count = 0  # Frame counter

# Ensure the output directory exists
output_dir = "training_set"
os.makedirs(output_dir, exist_ok=True)

print("Saving images to:", os.path.abspath(output_dir))

# Open video stream
vid = cv2.VideoCapture('http://172.20.10.3/xiao/Hi-Xiao-Ling')  # Open webcam capture

while True:
    ret, frame = vid.read()  # Get frame-by-frame
    
    if frame is not None:
        if dropped > 0:
            dropped = 0  # Reset drop counter
        
        # Display the video stream
        cv2.imshow('Video-44', frame)
        
        # Save only every 5th frame
        if frame_count % 20 == 0:
            frame_filename = os.path.join(output_dir, f"paper_{frame_count:04d}.jpg")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                print(f"Saved: {frame_filename}")
            else:
                print(f"Failed to save: {frame_filename}")
        
        frame_count += 1  # Always increment
        
        # Exit if 'q' is pressed
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        dropped += 1
        if dropped > 100:
            print("Server is down")
            break

# Release resources
vid.release()
cv2.destroyAllWindows()
print("Video stopped")