#!/bin/bash

# Construir la imagen
docker-compose -f ./docker-compose.yml build

# Ejecutar el servicio de entrenamiento
docker-compose -f ./docker-compose.yml run --rm entrenamiento

# Ejecutar el servicio de predicción
docker-compose -f ./docker-compose.yml run --rm prediccion
