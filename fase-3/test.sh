curl -X POST "http://localhost:8000/process/" \
    -F "train_file=@train.csv" \
    -F "predict_file=@test.csv" \
    -o predicts.csv
