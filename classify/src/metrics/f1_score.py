from sklearn.metrics import f1_score


def calculate_f1_score(true_labels, predicted_labels, average="weighted"):
    return f1_score(true_labels, predicted_labels, average=average)
