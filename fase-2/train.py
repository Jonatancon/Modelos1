import numpy as np
import pandas as pd
import os
import tensorflow_decision_forests as tfdf
import tensorflow as tf
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

def main(ruta_csv = '../fase-1/train.csv'):
    train_df = pd.read_csv(ruta_csv)
    preprocessed_train_df = preprocess(train_df)

    input_features = list(preprocessed_train_df.columns)
    input_features.remove("Ticket")
    input_features.remove("PassengerId")
    input_features.remove("Survived")
    #input_features.remove("Ticket_number")

    def tokenize_names(features, labels=None):
        """Divite the names into tokens. TF-DF can consume text tokens natively."""
        features["Name"] =  tf.strings.split(features["Name"])
        return features, labels

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df,label="Survived").map(tokenize_names)

    model = tfdf.keras.GradientBoostedTreesModel(
        verbose=0, # Very few logs
        features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
        exclude_non_specified_features=True, # Only use the features in "features"
        random_seed=1234,
    )
    model.fit(train_ds)
    model.summary()

    modelo_path = "./model"
    model.save(modelo_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo con TensorFlow DF")
    parser.add_argument("ruta_csv", type=str, help="Ruta del archivo CSV de entrenamiento")
    args = parser.parse_args()

    main(args.ruta_csv)