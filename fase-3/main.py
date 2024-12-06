import os
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse
import shutil
import subprocess

app = FastAPI()

# Directorios compartidos con los contenedores
UPLOAD_FOLDER = "./uploaded_files"
PREDICTIONS_FOLDER = "./predictions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)


@app.post("/process/")
async def process_files(
    train_file: UploadFile = File(..., description="Archivo CSV para entrenar el modelo"),
    predict_file: UploadFile = File(..., description="Archivo CSV para generar predicciones"),
):
    # Verificar extensiones de archivo
    for file in [train_file, predict_file]:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"El archivo {file.filename} no es un archivo CSV válido")

    # Guardar el archivo de entrenamiento
    train_file_path = os.path.join(UPLOAD_FOLDER, "train.csv")
    with open(train_file_path, "wb") as f:
        shutil.copyfileobj(train_file.file, f)

    # Guardar el archivo de predicción
    predict_file_path = os.path.join(UPLOAD_FOLDER, "test.csv")
    with open(predict_file_path, "wb") as f:
        shutil.copyfileobj(predict_file.file, f)

    # Ejecutar el proceso de entrenamiento
    try:
        subprocess.run(["docker-compose", "run", "entrenamiento"], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error durante el entrenamiento: {e}")

    # Ejecutar el proceso de predicción
    try:
        subprocess.run(["docker-compose", "run", "prediccion"], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    # Verificar que el archivo de predicciones exista
    prediction_file = os.path.join(PREDICTIONS_FOLDER, "predicts.csv")
    if not os.path.exists(prediction_file):
        raise HTTPException(status_code=500, detail="El archivo de predicciones no fue generado.")

    # Retornar el archivo de predicciones
    return FileResponse(prediction_file, media_type="text/csv", filename="predicts.csv")
