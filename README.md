# RPS_Project
 ENMGT 5400 Project 1 Code.

 The contents of the Local directory are all the files that are from my local (laptop).
 The contents of the ESP32S3 directory are all the files that are on my microcontroller.

 Most importantly, the rps_bmp.ipynb file contains the code for creating the
 keras CNN. The rps_streaming file is the python script that is run to receive
 the classification results and streaming information from the microcontroller.
 The streaming_server.py file on the microcontroller is what runs to capture
 bmp images, classify them, and send that information across the wifi network
 to the client. All passwords and addresses have been removed from the source
 code and the user must input their appropriate ones to run the code.
