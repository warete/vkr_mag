from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from os import path

from classification import Classification

def main() -> None:
    dataset = pd.read_csv(path.join('data', 'out_all.csv'))
    X, Y = dataset.drop('class', axis=1), dataset['class']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    print('shapes: x_train={}, x_test={}, y_train={}, y_test={}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    clf = Classification()

    for method_key, method in clf.methods.items():
        print('method', method['name'])
        clf.fit_model(method_key, x_train, y_train)
        y_pred = clf.predict_model(method_key, x_test)
    
        print(accuracy_score(y_test, y_pred))
        print(''.join(['-' for _ in range(15)]))


if __name__ == '__main__':
    main()
