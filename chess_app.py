import pandas as pd
import streamlit as st
import cv2
from time import time

st.markdown("# Chesspiece-Detection ♚ ♛")


# Camera setup
run = st.checkbox('Place your chessboard')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

# Initialize time variables
previous = time()
delta = 0

text_location = st.empty()
image_location = st.empty()

while run:
    # Get the current time, increase delta and update the previous variable
    current = time()
    delta += current - previous
    previous = current

     # Check if 3 seconds passed
    if delta > 5:
        # Show image
        text_location.text("New Picture")
        image_location.image(frame)
        # Reset the time counter
        delta = 0

    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(2)

