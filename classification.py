import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from os import path
import util

from classification import Classification


def classify_for_dataset(dataset) -> None:
    X, Y = dataset.drop(['class'] + ['t' + str(x) for x in range(36, 44)], axis=1, errors='ignore'), dataset['class']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

    print('shapes: x_train={}, x_test={}, y_train={}, y_test={}'.format(x_train.shape, x_test.shape, y_train.shape,
                                                                        y_test.shape))

    clf = Classification()

    for method_key, method in clf.methods.items():
        print('method', method['name'])
        clf.fit_model(method_key, x_train, y_train)
        y_pred = clf.predict_model(method_key, x_test)

        print('Точность классификации: {}'.format(util.toFixed(accuracy_score(y_test, y_pred), 4)))
        print('Чувствительность: {}'.format(util.toFixed(util.calculate_sensitivity(y_test, y_pred), 4)))
        print('Специфичность: {}'.format(util.toFixed(util.calculate_specificity(y_test, y_pred), 4)))
        print('Эффективность: {}'.format(util.toFixed(
            math.sqrt(util.calculate_specificity(y_test, y_pred) * util.calculate_specificity(y_test, y_pred)), 4)))
        print(''.join(['-' for _ in range(75)]))
    print(''.join(['/' for _ in range(75)]))
    print(''.join(['-' for _ in range(75)]))


def main() -> None:
    dataset = pd.read_csv(path.join('data', 'out_all.csv'))

    print('Правая + левая МЖ')
    classify_for_dataset(dataset)

    print('Левая + правая МЖ')
    classify_for_dataset(dataset.reindex(
        columns=['t' + str(i) for i in range(18, 36)] + ['t' + str(i) for i in range(18)] + ['class']
    ))

    print('Только левая МЖ')
    classify_for_dataset(dataset.drop(['t' + str(i) for i in range(18)], axis=1))

    print('Только правая МЖ')
    classify_for_dataset(dataset.drop(['t' + str(i) for i in range(18, 36)], axis=1))


if __name__ == '__main__':
    main()
