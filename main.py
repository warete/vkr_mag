from model import RtmModel, get_datasets_from_file
import time
import os

datasets = [
    {
        "name": "healthy",
        "file": "data_fit_with_d_rl_0.csv",
        "time": 0
    },
    {
        "name": "sick",
        "file": "data_fit_with_d_rl_1.csv",
        "time": 0
    },
]

time_results = []

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
#todo: load model
