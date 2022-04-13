from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class Classification():
    methods = {
        'svm': {
            'name': 'SVM',
            'clf': GridSearchCV(SVC(), {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [1, 10], 'gamma': ('scale', 'auto')}, cv=3),
        },
        'knn': {
            'name': 'k-ближайших соседей',
            'clf': GridSearchCV(neighbors.KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 8, 10], 'weights': ('uniform', 'distance'), 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}, cv=3),
        },
        'decisiontreeclassifier': {
            'name': 'Дерево решений',
            'clf': GridSearchCV(DecisionTreeClassifier(), {'criterion': ('gini', 'entropy'), 'splitter': ('best', 'random'), 'max_depth': [1, 2, 3, 4, 5]}, cv=3),
        },
        'logisticRegression': {
            'name': 'Логистическая регрессия',
            'clf': GridSearchCV(LogisticRegression(random_state=0, max_iter=10000), {}, cv=3),
        },
    }

    def fit_model(self, model_key: str, x, y) -> None:
        if self.methods[model_key]:
            self.methods[model_key]['clf'].fit(x, y)

    def predict_model(self, model_key: str, x) -> list:
        if self.methods[model_key]:
            return self.methods[model_key]['clf'].predict(x)
        else:
            return []
