curl -X POST "http://localhost:8000/process/" ^
    -F "train_file=@C:\ruta\completa\train.csv" ^
    -F "predict_file=@C:\ruta\completa\test.csv" ^
    -o predicts.csv
