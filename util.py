from sklearn.metrics import confusion_matrix


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def calculate_sensitivity(y_test, y_pred) -> float:
    cm1 = confusion_matrix(y_test, y_pred)
    return cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])


def calculate_specificity(y_test, y_pred) -> float:
    cm1 = confusion_matrix(y_test, y_pred)
    return cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
