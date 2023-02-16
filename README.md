# Chesspiece-Detection


## Project structure
```bash
├── Pipfile
├── Pipfile.lock
├── README.md
├── __pycache__
├── check_annotations.ipynb
├── data
│   ├── checked_annot                                   # output of check_annotations.ipynb
│   │   ├── bottom_annot.csv
│   │   ├── bottom_annot_1602_downwards.csv
│   │   └── top_184_anot.csv
│   ├── chess_new.csv
│   ├── chess_new.p                                     # all correctly annotated chess data
│   ├── chess_resized.p                                 # all resized correctly annotated data 
│   ├── df_train.pickle
│   ├── final_data                                      # new data
│   │   ├── test
│   │   │   ├── 04b4ee23-00000062_jpg.rf.88154d532043be1e7895d6c96bb2076c.jpg.jpeg
│   │   │   ├── 04dccafa-00000065_jpg.rf.fe1c7911799a247d7c8637a6fa60a78e.jpg.jpeg
│   │   │   ├── ...
│   │   │   ├── chess_test_df.p
│   │   ├── train
│   │   │   ├── 00ddcec9-00000008_png.rf.d9ba4a5b95898a8096842974a6ba2ed4.jpg.jpeg
│   │   │   ├── 036f8818-00000133_jpg.rf.6612acf9df55cfe0741012f2b52ac666.jpg.jpeg
│   │   │   ├── ...
│   │   │   ├── chess_train_df.p
│   │   └── val
│   │       ├── 06dca0c7-00000114_jpg.rf.67e7267c9565083a492d2c2dfac12f04.jpg.jpeg
│   │       ├── 073210af-00000014_jpg.rf.7b2177680f2dd8aa4fa332264a4a41f5.jpg.jpeg
│   │       ├── ...
│   │       ├── chess_val_df.p
│   └── roboflow_data                                   # original data
│       ├── test
│       │   ├── 00043390-00000150_png.rf.a9073878ea93c844f6f1ee690e65fa7b.jpg
│       │   ├── 0a5d3b7c-00000075_jpg.rf.7ddce7f9033f988f1170c20dd6c1a137.jpg
│       │   ├── ...
│       │   ├── _annotations.csv
│       ├── train
│       │   ├── 00ddcec9-00000008_png.rf.d9ba4a5b95898a8096842974a6ba2ed4.jpg
│       │   ├── 014f284f-00000046_jpg.rf.02a8b7629b7d052a95972da3f302d338.jpg
│       │   ├── ...
│       │   ├── _annotations.csv
│       └── valid
│           ├── 00026ec4-00000116_jpg.rf.0d973f65d1a9bc24ed144e8b7ff63862.jpg
│           ├── 00cda564-00000058_jpg.rf.ca780bf8e28ef2e7b6cb9d3dfe0d689e.jpg
│           ├── ...
│           ├── _annotations.csv
├── data_preprocessing.ipynb
└── model_build.ipynb
```