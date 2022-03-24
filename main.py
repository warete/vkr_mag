from model import RtmModel, get_datasets_from_file
import time
import os

datasets = [
    {
        "name": "healthy",
        "file": "data_fit_with_d_rl_0.csv"
    },
    {
        "name": "sick",
        "file": "data_fit_with_d_rl_1.csv"
    },
]

for dataset in datasets:
    train_dataset, test_dataset = get_datasets_from_file(os.path.join('data', dataset['file']))

    model = RtmModel(dataset['name'])

    model.compile(0.005)
    start_t = time.time()
    model.fit(train_dataset, validation_data=test_dataset, epochs=1000)
    end_t = time.time()
    print('Finish time: ', end_t - start_t)

# model.save(os.path.join('model_checkpoints', 'rtm'))