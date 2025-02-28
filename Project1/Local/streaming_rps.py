import cv2
import numpy as np
import struct
import requests

def main():
    # Set the URL for the ESP32 Camera stream
    url = 'http://0.0.0.0/xiao/Hi-Xiao-Ling'  # Replace with your actual server URL

    classes = ["Paper", "Rock", "Scissors"]

    # Initialize a session to handle the request
    with requests.Session() as session:
        # Send a GET request to the server
        response = session.get(url, stream=True)

        # Check if the request was successful
        if response.status_code == 200:
            print("Connected to the camera stream.")

            # Read the response in chunks
            boundary = b'--kaki5'  # The boundary that separates the frames
            image_data = b''
            running = True  # Flag to control the main loop

            while running:
                for chunk in response.iter_content(chunk_size=1024):
                    image_data += chunk
                    # Check if we have enough data to find a full image
                    if boundary in image_data:
                        # Split the data into frames
                        frames = image_data.split(boundary)
                        for frame in frames[:-1]:  # Ignore the last chunk (incomplete frame)
                            if b'Content-Type: image/bmp' in frame:
                                # Extract the BMP image data
                                header_index = frame.index(b'\r\n\r\n') + 4  # Skip the header
                                bmp_data = frame[header_index:-4]  # Get the BMP data
                                classification_bytes = frame[-4:] # Last 4 bytes of the message

                                predicted_label = struct.unpack('i', classification_bytes)[0]
                                if (predicted_label in range(len(classes))):
                                    print(classes[predicted_label])
                                else:
                                    print("invalid prediction")
                                running = display_image(bmp_data)

                        # Keep only the last (incomplete) frame in the buffer
                        image_data = frames[-1]  # This might contain partial image data
                        
                    # Check if we should exit the loop after processing
                    if not running:
                        break  # Exit the for loop if running is False
        else:
            print("Failed to connect to the camera stream.")

def display_image(bmp_data):
    # Convert the BMP data to a NumPy array
    nparr = np.frombuffer(bmp_data, np.uint8)

    # Read the image from the NumPy array
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Check if the image was decoded successfully
    if image is not None:
        cv2.imshow('ESP32 Camera Stream', image)  # Display the image
        key = cv2.waitKey(1)  # Wait for a key press
        if key & 0xFF == ord('q'):  # Press 'q' to quit
            cv2.destroyAllWindows()
            return False  # Indicate to stop running
    return True  # Continue running

if __name__ == '__main__':
    main()
