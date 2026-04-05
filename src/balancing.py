import numpy as np

def compute_class_weights(y):
    """
    Computes inverse-frequency class weights.
    Minority class ('1') gets a higher weight, majority class ('0') gets a lower one.
    No data is removed or synthesized — only the loss contribution per sample is scaled.

    Formula: weight_c = total_samples / (n_classes * count_c)
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)

    weights = {int(c): total / (n_classes * count) for c, count in zip(classes, counts)}
    return weights


def get_sample_weights(y, class_weights):
    """
    Expands class weights into a per-sample weight array.
    Each sample gets the weight of its class.
    """
    return np.array([class_weights[int(label)] for label in y])
