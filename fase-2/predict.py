import numpy as np
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import keras
import argparse

def preprocess(df):
    df = df.copy()
    
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
    return df

def main(ruta_modelo='./model', ruta_test='../fase-1/test.csv', ruta_resultado='./predicts.csv'):
    model = keras.layers.TFSMLayer(ruta_modelo, call_endpoint='serving_default')

    print('====================================================')
    print(model)
    print('====================================================')


    serving_df = pd.read_csv(ruta_test)
    preprocessed_serving_df = preprocess(serving_df)

    def tokenize_names(features, labels=None):
        """Divite the names into tokens. TF-DF can consume text tokens natively."""
        features["Name"] =  tf.strings.split(features["Name"])
        return features, labels
    
    serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_serving_df).map(tokenize_names)

    def prediction_to_kaggle_format(model, threshold=0.5):
        features_tensor = serving_ds.map(lambda x, _: x)
        proba_survive = model(features_tensor)[:,0]

        return pd.DataFrame({
            "PassengerId": serving_df["PassengerId"],
            "Survived": (proba_survive >= threshold).astype(int)
        })
    
    result = prediction_to_kaggle_format(model)
    result.to_csv(ruta_resultado, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera predicciones con un modelo entrenado.")
    parser.add_argument("ruta_modelo", type=str, help="Ruta al modelo entrenado")
    parser.add_argument("ruta_test", type=str, help="Ruta al archivo de prueba")
    parser.add_argument("ruta_resultado", type=str, help="Ruta para guardar el resultado en CSV")

    args = parser.parse_args()

    main(args.ruta_modelo, args.ruta_test, args.ruta_resultado)