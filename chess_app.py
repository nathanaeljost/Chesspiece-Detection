import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from time import time

st.markdown("# Chesspiece-Detection ♚ ♛")


# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='data/detect_28.03.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.write(output_details)

# Define the category index
category_index = {1: {'id': 1, 'name': 'black-bishop'},
                  2: {'id': 2, 'name': 'black-king'},
                  3: {'id': 3, 'name': 'black-queen'},
                  4: {'id': 4, 'name': 'black-rook'},
                  5: {'id': 5, 'name': 'white-bishop'},
                  6: {'id': 6, 'name': 'white-knight'},
                  7: {'id': 7, 'name': 'white-king'},
                  8: {'id': 8, 'name': 'white-queen'},
                  9: {'id': 9, 'name': 'white-pawn'},
                  10: {'id': 10, 'name': 'white-rook'},
                  11: {'id': 11, 'name': 'black-pawn'},
                  12: {'id': 12, 'name': 'black-knight'}}


# Camera setup
run = st.checkbox('Place your chessboard')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Initialize time variables
previous = time()
delta = 0

# Initialize chess figures
chess_figures = {'black-bishop':2, 'black-king':1, 'black-queen':1, 'black-rook':2,
'white-bishop':2, 'white-knight':2, 'white-king':1, 'white-queen':1,'black-knight':2, 
'black-pawn':8, 'white-pawn':8, 'white-rook':2}

text_location = st.empty()
image_location = st.empty()

while run:
    # Get the current time, increase delta and update the previous variable
    current = time()
    delta += current - previous
    previous = current

     # Check if 3 seconds passed
    if delta > 3:
        # Show image
        text_location.text("New Picture")
        image_location.image(frame)
        # Reset the time counter
        delta = 0

    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare input data
    resized_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    boxes = output_data[0]
    st.write(output_data)
    st.write(boxes)

    # Draw the detection boxes onto the image
    if boxes.any():
        for box in boxes:
            # Get coordinates of the box
            ymin, xmin, ymax, xmax = box

            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw the box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # Display the annotated image
    FRAME_WINDOW.image(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(2)

