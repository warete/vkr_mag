def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def calculate_sensitivity(y_test, y_pred) -> float:
    sick_test_cnt = len(y_test[y_test == 1])
    sick_pred_cnt = len(y_pred[y_pred == 1])
    return sick_pred_cnt / (sick_pred_cnt + abs(sick_test_cnt - sick_pred_cnt))


def calculate_specificity(y_test, y_pred) -> float:
    healthy_test_cnt = len(y_test[y_test == 0])
    healthy_pred_cnt = len(y_pred[y_pred == 0])
    return healthy_pred_cnt / (healthy_pred_cnt + abs(healthy_test_cnt - healthy_pred_cnt))
