# Titanic - Machine Learning from Disaster
## Fase 1: Introducción al Desafío
### Descripción del Desafío

*El hundimiento del RMS Titanic es uno de los desastres marítimos más conocidos de la historia. Ocurrió el 15 de abril de 1912, cuando el Titanic, considerado "insumergible", chocó con un iceberg durante su viaje inaugural. De los 2.224 pasajeros y tripulantes a bordo, solo 722 personas sobrevivieron, dejando un saldo trágico de 1.502 víctimas.*

### Objetivo del Desafío

**El objetivo principal de este reto es construir un modelo de aprendizaje automático que prediga qué tipo de personas tenían más probabilidades de sobrevivir al desastre del Titanic, utilizando los datos proporcionados de los pasajeros. Estos datos incluyen información como el nombre, la edad, el sexo, la clase socioeconómica, entre otros.**

## Fase 2: Docker

Se crean los scripts solicitados de `predict.py` y `train.py`, además se dokerisa la aplicación.

El proyecto para esta fase presenta un problema. El modelo exportado del `train` para usar en el `predict` presenta problemas por la clase de librerías que se usa. Se cambiará este enfoque para la próxima entrega para permitir el funcionamiento correcto.

## Fase 3: API

Se crea un script donde se expone un end-point, el cual recibirá dos archivos, test y train, para así utilizar el script de la fase 2 para enseñar al modelo y poder hacer las predicciones.

Se modificó el Dockerfile para que instalara las dependencias esperadas y el docker-compose para que expusiera el servicio de la API, de esta manera solo se tiene que ejecutar dos comandos:

    1) docker-compose build
    2) docker-compose up

Después de tener el contenedor arriba, se puede probar el servicio ejecutando el script **test.sh** *(Linux)* o **test.bat** *(Windows)* dependiendo del sistema operativo que se tenga.

Por último, se descargará un archivo **predict.csv** con las predicciones hechas por el modelo.


