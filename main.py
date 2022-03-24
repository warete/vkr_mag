from model import RtmModel, get_datasets_from_file, load_model
import time
import os
import sys

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
        
        train_dataset, test_dataset = get_datasets_from_file(os.path.join('data', dataset['file']))

        model = RtmModel(dataset['name'])

        model.compile(0.005)
        start_t = time.time()
        model.fit(train_dataset, validation_data=test_dataset, epochs=1000)
        end_t = time.time()
        datasets[k]['time'] = end_t - start_t
        model.save(os.path.join('model_checkpoints', dataset['name']))

    for dataset in datasets:
        print('Finish time', dataset['name'], ':', end_t - start_t)

def predict_model_data(datasets):
    """Достраивает парную МЖ для модельных данных"""
    for k in range(len(datasets)):
        dataset = datasets[k]

        model_path = os.path.join('model_checkpoints', dataset['name'])

        model = load_model(model_path)
