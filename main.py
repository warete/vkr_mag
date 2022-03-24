from sklearn.preprocessing import scale
from model import RtmModel, get_datasets_from_file, load_model, get_scaler, get_scaled_data
import pandas as pd
import numpy as np
import time
import os

datasets = [
    {
        "name": "healthy",
        "file": "data_fit_with_d_rl_0.csv",
        "time": 0,
        "model_data_file": "model_healthy.csv",
    },
    {
        "name": "sick",
        "file": "data_fit_with_d_rl_1.csv",
        "time": 0,
        "model_data_file": "model_sick.csv",
    },
]


def fit_datasets(datasets):
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

        model.compile(0.005)
        start_t = time.time()
        model.fit(train_dataset, validation_data=test_dataset, epochs=1000)
        end_t = time.time()
        datasets[k]['time'] = end_t - start_t
        model.save(os.path.join('model_checkpoints', dataset['name']))

    for dataset in datasets:
        print('Finish time', dataset['name'], ':', datasets[k]['time'])


def predict_model_data(datasets):
    """Достраивает парную МЖ для модельных данных"""
    for k in range(len(datasets)):
        dataset = datasets[k]

        model_path = os.path.join('model_checkpoints', dataset['name'])

        model = load_model(model_path)

        scaler = get_scaler()
        test_X, source_x = get_scaled_data(os.path.join(
            'data', dataset['model_data_file']), scaler)

        pred_test_X = model.predict(test_X)
        pred_test_X_unscaled = scaler.inverse_transform(pred_test_X)
        pred_df = pd.DataFrame(np.hstack((source_x, pred_test_X_unscaled)), columns=[
                               't' + str(key) for key in range(36)])
        predict_result_file = os.path.join(
            'data', 'out_' + dataset['name'] + '.csv')
        pred_df.to_csv(predict_result_file, sep=',',
                       encoding='utf-8', index=False)
        print('Predict data for', dataset['name'], 'saved to', predict_result_file)


if __name__ == '__main__':
    fit_datasets(datasets)
    predict_model_data(datasets)
