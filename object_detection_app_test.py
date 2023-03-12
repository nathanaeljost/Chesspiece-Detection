import re
import cv2
import numpy as np
import tensorflow as tf

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

labels = {1: {'id': 1, 'name': 'black-bishop'},
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


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image - 255) / 255, axis=0)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    # Get all output details
    scores = get_output_tensor(interpreter, 0)
    boxes = get_output_tensor(interpreter, 1)
    count = int(get_output_tensor(interpreter, 2))
    classes = get_output_tensor(interpreter, 3)

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
    interpreter = tf.lite.Interpreter(model_path='data/detect_new.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 320))
        res = detect_objects(interpreter, img, 0.25)
        print(res)

        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1, xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(frame, labels[int(result['class_id'])]['name'], (xmin, min(ymax, CAMERA_HEIGHT - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()