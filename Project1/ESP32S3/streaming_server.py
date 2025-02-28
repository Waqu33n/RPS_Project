
# The MIT License (MIT)
#
# Copyright (c) Sharil Tumin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#-----------------------------------------------------------------------------

# run this on ESP32 Camera

import esp
from Wifi import Sta
import array
import struct
import socket as soc
from time import sleep
from camera import FrameSize, PixelFormat, Camera
import emlearn_cnn_fp32 as emlearn_cnn
from image_preprocessing import strip_bmp_header, resize_96x96_to_32x32

# Model file name
MODEL = 'final_model.tmdl'

hdr = {
  # live stream -
  # URL: /live
  'stream': """HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=kaki5
Connection: keep-alive
Cache-Control: no-cache, no-store, max-age=0, must-revalidate
Expires: Thu, Jan 01 1970 00:00:00 GMT
Pragma: no-cache""",
  # live stream -
  # URL:
  'frame': """--kaki5
Content-Type: image/bmp"""}

esp.osdebug(None)   # turn off debugging log. Uncomment to show debugging log

UID = const('xiao')          # authentication user
PWD = const('Hi-Xiao-Ling')  # authentication password

# Set Camera Parameters
CAMERA_PARAMETERS = {
    "data_pins": [15,17,18,16,14,12,11,48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "frame_size": FrameSize.R96X96,
    "pixel_format": PixelFormat.GRAYSCALE
}

# Initialize Camera
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera ready?: ", cam)

# connect to access point
sta = Sta()              # Station mode (i.e. need WiFi router)
sta.wlan.disconnect()    # disconnect from previous connection
AP = const("YOUR-SSID") # Your SSID
PW = const('password') # Your password
sta.connect(AP, PW) # connet to dlink
sta.wait()

# load model
model = None
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())
    model = emlearn_cnn.new(model_data)
# create probabilities array
out_length = model.output_dimensions()[0]
probabilities = array.array('f', (-1 for _ in range(out_length)))


# wait for WiFi
con = ()
for i in range(5):
    if sta.wlan.isconnected():con=sta.status();break
    else: print("WIFI not ready. Wait...");sleep(2)
else:
    print("WIFI not ready")

if con and cam: # WiFi and camera are ready
   if cam:
       cam.set_frame_size(0)
       cam.set_special_effect(2)
   if con:
     # TCP server
     port = 80
     addr = soc.getaddrinfo('0.0.0.0', port)[0][-1]
     s = soc.socket(soc.AF_INET, soc.SOCK_STREAM)
     s.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
     s.bind(addr)
     s.listen(1)
     # s.settimeout(5.0)
     while True:
        cs, ca = s.accept()   # wait for client connect
        print('Request from:', ca)
        w = cs.recv(200) # blocking
        (_, uid, pwd) = w.decode().split('\r\n')[0].split()[1].split('/')
        # print(_, uid, pwd)
        if not (uid==UID and pwd==PWD):
           print('Not authenticated')
           cs.close()
           continue
        # We are authenticated, so continue serving
        cs.write(b'%s\r\n\r\n' % hdr['stream'])
        pic=cam.capture
        put=cs.write
        hr=hdr['frame']
        while True:
           # once connected and authenticated just send the jpg data
           # client use HTTP protocol (not RTSP)
           try:
              # Classify image
              put(b'%s\r\n\r\n' % hr)
              img = pic()
              img_resize = resize_96x96_to_32x32(img)
              img_strip = strip_bmp_header(array.array('B', img_resize))
              model.run(img_strip, probabilities)

              # Process the probabilities and display results
              predicted_class = -1
              max_prob = -1
              for i in range(len(probabilities)):
                if probabilities[i] > max_prob:
                    max_prob = probabilities[i]
                    predicted_class = i
              # Send image to client
              put(img)
              put(b'\r\n')  # send and flush the send buffer
              # Send prediction to client
              put(struct.pack('i',predicted_class))
           except Exception as e:
              print('TCP send error', e)
              cs.close()
              break
else:
   if not con:
      print("WiFi not connected.")
   if not cam:
      print("Camera not ready.")
   print("System not ready. Please restart")

print('System aborted')
