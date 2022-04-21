from sklearn.preprocessing import scale
from model import RtmModel, get_datasets_from_file, load_model, get_scaler, get_scaled_data, get_data_from_file, \
    scale_data
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import shutil

from util import toFixed

datasets = [
    {
        "name": "healthy",
        "file": "data_fit_with_d_rl_0.csv",
        "time": 0,
        "model_data_file": "model_healthy.csv",
        "learning_rate": 0.05,
        "learning_epochs": 500,
        "class": 0,
    },
    {
        "name": "sick",
        "file": "data_fit_with_d_rl_1.csv",
        "time": 0,
        "model_data_file": "model_sick.csv",
        "learning_rate": 0.03,
        "learning_epochs": 700,
        "class": 1,
    },
]

def fit_datasets(datasets, need_tensorboard: bool):
    """Обучает модели на реальных данных"""
    for k in range(len(datasets)):
        dataset = datasets[k]

        model_path = os.path.join('model_checkpoints', dataset['name'])

        if os.path.isdir(model_path):
            continue

        scaler = get_scaler()
        train_dataset, test_dataset = get_datasets_from_file(
            os.path.join('data', dataset['file']), scaler)

        model = RtmModel(dataset['name'])

        model.compile(dataset['learning_rate'])

        fit_callbacks = []
        if need_tensorboard:
            logs_dir_name = 'logs_'+model.name
            if os.path.isdir(logs_dir_name):
                shutil.rmtree(logs_dir_name)
            
            fit_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logs_dir_name))
        start_t = time.time()
        model.fit(train_dataset, validation_data=test_dataset, epochs=dataset['learning_epochs'], callbacks=fit_callbacks)
        end_t = time.time()
        datasets[k]['time'] = end_t - start_t
        model.save(os.path.join('model_checkpoints', dataset['name']))

    for dataset in datasets:
        print('Finish time', dataset['name'], ':', datasets[k]['time'])


def predict_model_data(datasets):
    """Достраивает парную МЖ для модельных данных"""

    all_data = None
    for k in range(len(datasets)):
        dataset = datasets[k]

        model_path = os.path.join('model_checkpoints', dataset['name'])

        model = load_model(model_path)

        file_data = get_data_from_file(os.path.join('data', dataset['model_data_file']))

        scaler18 = get_scaler()
        scaler4 = get_scaler()

        scale_data(file_data.values[:, :4], scaler4)

        test_X = scale_data(file_data.values, scaler18)

        pred_test_X = model.predict(test_X)
        pred_test_X_all_t_unscaled = scaler18.inverse_transform(pred_test_X[:, :18])
        pred_test_X_a_t_unscaled = scaler4.inverse_transform(pred_test_X[:, 18:])
        pred_test_X_unscaled = np.hstack((pred_test_X_all_t_unscaled, pred_test_X_a_t_unscaled))
        pred_text_X_formatted = []
        for row in pred_test_X_unscaled:
            pred_text_X_formatted.append([toFixed(x, 2) for x in row])
        
        result_rows = np.hstack((file_data.values, pred_text_X_formatted))
        result_rows_with_class = np.append(result_rows, np.array([[dataset['class']] for _ in range(len(result_rows))]), axis=1)
        if all_data is None:
            all_data = result_rows_with_class
        else:
            all_data = np.concatenate([all_data, result_rows_with_class])
        pred_df = pd.DataFrame(result_rows_with_class, columns=[
                               't' + str(key) for key in range(36)] + ['mw_a_1', 'mw_a_2', 'ir_a_1', 'ir_a_2'] + ['class'])
        predict_result_file = os.path.join(
            'data', 'out_' + dataset['name'] + '.csv')
        pred_df.to_csv(predict_result_file, sep=',',
                       encoding='utf-8', index=False)
        print('Predict data for', dataset['name'], 'saved to', predict_result_file)
    pred_all_df = pd.DataFrame(all_data, columns=[
                               't' + str(key) for key in range(36)] + ['mw_a_1', 'mw_a_2', 'ir_a_1', 'ir_a_2'] + ['class'])
    predict_all_result_file = os.path.join(
        'data', 'out_all.csv')
    pred_all_df.to_csv(predict_all_result_file, sep=',',
                    encoding='utf-8', index=False)
    print('Predict data all saved to', predict_all_result_file)
    print('finished')


if __name__ == '__main__':
    fit_datasets(datasets, True)
    predict_model_data(datasets)
