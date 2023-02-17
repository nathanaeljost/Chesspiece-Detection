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
    FRAME_WINDOW.image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(2)

