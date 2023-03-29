import re
import time
import cv2
import numpy as np
import tensorflow as tf

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960


labels = { 1 : 'black-bishop',
           2 : 'black-king',
           3 : 'black-queen',
           4 : 'black-rook',
           5 : 'white-bishop',
          6 : 'white-knight',
          7 : 'white-king',
          8 : 'white-queen',
           9 : 'white-pawn',
           10 : 'white-rook',
           11 : 'black-pawn',
          12 : 'black-knight'}


#model_path = '/data/detect_28.03.tflite'

# Load the labels into a list
classes = ['???'] * 12 

for label_id, label_name in labels.items():
   classes[label_id-1] = label_name

print(classes[2])
#classes = [labels[i]['name'] for i in range(1, 12)]

# Define a list of colors for visualization
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

def preprocess_image(img, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  #img = tf.io.read_file(image_path)
  #img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  resized_img = tf.cast(resized_img, dtype=tf.uint8)
  return resized_img, original_image


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""

  signature_fn = interpreter.get_signature_runner()

  # Feed the input image to the model
  output = signature_fn(images=image)

  # Get all outputs from the model
  count = int(np.squeeze(output['output_0']))
  scores = np.squeeze(output['output_1'])
  classes = np.squeeze(output['output_2'])
  boxes = np.squeeze(output['output_3'])

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def main():
    interpreter = tf.lite.Interpreter(model_path='data/model_old_28.03.tflite')
    interpreter.allocate_tensors()
    #_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

      # Load the input image and preprocess it
        preprocessed_image, original_image = preprocess_image(
            frame,
            (input_height, input_width)
          )

        # Run object detection on the input image
        results = detect_objects(interpreter, preprocessed_image, threshold=0.25)
        print(results)

        # Plot the detection results on the input image
        #original_image_np = original_image.numpy().astype(np.uint8)
        for obj in results:
          # Convert the object bounding box from relative coordinates to absolute
          # coordinates based on the original image resolution
          ymin, xmin, ymax, xmax = obj['bounding_box']
          xmin = int(max(1, xmin * CAMERA_WIDTH))
          xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
          ymin = int(max(1, ymin * CAMERA_HEIGHT))
          ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

          # Find the class index of the current object
          class_id = int(obj['class_id'])
          #print(COLORS)
          # Draw the bounding box and label on the image
          #print(class_id)
          color = [int(c) for c in COLORS[class_id]]
          
          cv2.rectangle(frame, (xmin, ymin-100), (xmax, ymax-100), color, 2)
          # Make adjustments to make the label visible for all objects
          y = ymin - 15 if ymin - 15 > 15 else ymin + 15
          label = "{}: {:.0f}%".format(classes[class_id], obj['score'] * 100)
          cv2.putText(frame, label, (xmin, y),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Chess figure detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()