from pydoc import source_synopsis
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split

class RtmModel(tf.keras.Model):
    name = ''

    def __init__(self, name):
        super().__init__()

        self.name = name

        self.inputs = tf.keras.layers.Input(name='values18', shape=(18,), dtype='float32')

        self.dp1 = tf.keras.layers.Dropout(0.1)
        self.dense1 = tf.keras.layers.Dense(36, activation='sigmoid')
        self.dp2 = tf.keras.layers.Dropout(0.1)
        self.dp3 = tf.keras.layers.Dropout(0.1)
        self.dense2 = tf.keras.layers.Dense(18)

    def call(self, inputs, training):
        x = self.dp1(inputs)
        x = self.dense1(x)
        x = self.dp2(x)
        x = self.dp3(x)
        return self.dense2(x)
    
    def compile(self, learning_rate):
        super().compile(optimizer=tf.keras.optimizers.SGD(learning_rate), loss='MSE')

def get_scaler():
    return StandardScaler()

def get_scaled_data(file_path, scaler):
    df = pd.read_csv(file_path)
    source_values = df.values

    #data preprocessing
    scaler.fit(source_values)
    return scaler.transform(source_values), source_values

def get_datasets_from_file(file_path, scaler):
    source_values = get_scaled_data(file_path, scaler)

    #train/test split
    np.random.shuffle(source_values)
    train, test = train_test_split(source_values, test_size=0.3)
    train_X, train_y = train[:, :18], train[:, 18:]
    test_X, test_y = test[:, :18], test[:, 18:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(1000).repeat(10)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(1000).repeat(10)

    return train_dataset, test_dataset

def load_model(file):
    return tf.keras.models.load_model(file)
