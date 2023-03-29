# Chesspiece-Detection

## Notes
  * `1_eda_data_preprocessing.ipynb` - eda and data preprocessing. Second part uses annotations csv from `2_check_annotations.ipynb`. Output: final data in data/final_data/ and `data/final_data/test_df.csv` as well as `data/final_data/train_df.csv`
  * `2_check_annotations.ipynb` - uses `df_train.pickle` (available on request or output of `1_eda_data_preprocessing.ipynb`) provides annotation function that helps to label if annotations for an image are correct or not. Output: csv with that can be used to filter images based on correct or incorrect annotation.
  * `3_TFOD_detection_model.ipynb` - Modeling with TensorFlow Object Detection API. Caution: Notebook was created and run in Google Colab. The model uses `data/final_data/test_df.csv` and `data/final_data/train_df.csv` in cell number 8 that are run from private Google drive. Adjust path accordingly to be able to run notebook. Output: `model.tflite`
  * `4_object_detection_app.py` - Uses model from `data/model_x` to predict from live camera input. Threshold for detection to be adjusted in line 94.

## Project structure
```bash
.
|-- 1_eda_data_preprocessing.ipynb
|-- 2_check_annotations.ipynb
|-- 3_TFOD_detection_model.ipynb
|-- 4_object_detection_app.py
|-- Pipfile
|-- Pipfile.lock
|-- README.md
|-- archive
|   |-- Object_detection_tflite.ipynb
|   |-- TFOD_training_and_detection.ipynb
|   |-- TFOD_training_and_detection_whole_data.ipynb
|   |-- data_preprocessing_whole_data.ipynb
|   |-- generate_tfrecord.py
|   |-- new_detect.py
|   |-- object_detection_app_test.py
|   `-- static_detection.py
|-- data
|   |-- checked_annot
|   |-- chess_new.csv
|   |-- chess_whole.p
|   |-- detect.tflite
|   |-- detect_18.03.tflite
|   |-- detect_19.03.tflite
|   |-- detect_28.03.tflite
|   |-- detect_new.tflite
|   |-- final_csv
|   |-- final_data_whole
|   |-- final_data_whole.zip
|   |-- model.tflite
|   |-- model_new_29.03.tflite
|   |-- model_old_28.03.tflite
|   |-- roboflow_data
|   |-- saved_model
|   |-- ssd_resnet50.tflite
|   |-- test_1.jpeg
|   |-- test_2.jpeg
|   |-- test_3.jpeg
|   |-- test_4.jpeg
|   `-- tfl_data.csv
`-- streamlit_chess_app.py
```
