from camera import FrameSize, PixelFormat, Camera
import emlearn_cnn_fp32 as emlearn_cnn
from image_preprocessing import strip_bmp_header, resize_96x96_to_32x32
import array
import esp
import gc

# Model file
MODEL = 'final_model.tmdl'

# set camera parameters
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

# Init Camera
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)

print("camera initialized")

# 3 classification test images
paper1 = 'paper_0020_rot0.bmp'
rock1 = 'rock_0840_rot225.bmp'
scissor1 = 'scissor_0240_rot180.bmp'

# Load test images
with open(paper1, "rb") as f:
        paper_data = f.read()
with open(rock1, "rb") as f:
        rock_data = f.read()
with open(scissor1, "rb") as f:
        scissor_data = f.read()

print("loaded bmp")
# load model
model = None
with open(MODEL, 'rb') as f:
    model_data = array.array('B', f.read())
    gc.collect()
    model = emlearn_cnn.new(model_data)
    
# Create probabilities array
out_length = model.output_dimensions()[0]
probabilities = array.array('f', (-1 for _ in range(out_length)))

# Resize Image and classify
paper_resize = resize_96x96_to_32x32(paper_data)
paper_strip = strip_bmp_header(array.array('B', paper_resize))
try:
    model.run(paper_strip, probabilities)
except Exception as e:
    print("Error during model execution:", e)

# Process the probabilities and display results
predicted_class = -1
max_prob = -1
for i in range(len(probabilities)):
    if probabilities[i] > max_prob:
        max_prob = probabilities[i]
        predicted_class = i
print(f'Probabilities: {probabilities}')
print(f'Predicted class: {predicted_class}, Confidence: {max_prob:.2f}')

# Process the probabilities and display results
rock_resize = resize_96x96_to_32x32(rock_data)
rock_strip = strip_bmp_header(array.array('B', rock_resize))
try:
    model.run(rock_strip, probabilities)
except Exception as e:
    print("Error during model execution:", e)

# Process the probabilities and display results
predicted_class = -1
max_prob = -1
for i in range(len(probabilities)):
    if probabilities[i] > max_prob:
        max_prob = probabilities[i]
        predicted_class = i
print(f'Probabilities: {probabilities}')
print(f'Predicted class: {predicted_class}, Confidence: {max_prob:.2f}')

# Process the probabilities and display results
scissor_resize = resize_96x96_to_32x32(scissor_data)
scissor_strip = strip_bmp_header(array.array('B', scissor_resize))
try:
    model.run(scissor_strip, probabilities)
except Exception as e:
    print("Error during model execution:", e)

# Process the probabilities and display results
predicted_class = -1
max_prob = -1
for i in range(len(probabilities)):
    if probabilities[i] > max_prob:
        max_prob = probabilities[i]
        predicted_class = i
print(f'Probabilities: {probabilities}')
print(f'Predicted class: {predicted_class}, Confidence: {max_prob:.2f}')


cam.deinit()

