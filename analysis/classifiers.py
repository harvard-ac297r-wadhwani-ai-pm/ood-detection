import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report, confusion_matrix

def get_auroc(scores_id, scores_ood):
    '''
    Assumes scores_id and scores_ood are numpy arrays of anomaly detection
    scores for ID and OOD samples, respectively. Returns a single AUC value.
    '''
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)


def get_f1_maximizing_threshold(scores_id, scores_ood):
    '''
    Assumes scores_id and scores_ood are numpy arrays of anomaly detection
    scores for ID and OOD samples, respectively. Returns a tuple containing an
    anomaly detection threshold that maximizes the macro-averaged F1 score, and
    the maximum macro-averaged F1 score itself.
    '''
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    f1_scores = []
    for threshold in thresholds:
        pred_labels = 1.0*(scores > threshold)
        f1_scores.append(f1_score(labels, pred_labels, average='macro'))
    f1_scores = np.array(f1_scores)
    return thresholds[np.argmax(f1_scores)], np.max(f1_scores)


def get_classification_report(scores_id, scores_ood, threshold):
    '''
    Assumes scores_id and scores_ood are numpy arrays of anomaly detection
    scores for ID and OOD samples, respectively. Returns sklearn classification
    report and confusion matrix for the user-supplied anomaly detection
    threshold (typically determined with get_f1_maximizing_threshold above).
    '''
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    pred_labels = 1.0*(scores > threshold)
    class_report = classification_report(labels, pred_labels, digits=3, target_names=['OOD: 0', 'ID: 1'])
    conf_matrix = confusion_matrix(labels, pred_labels)
    return class_report, conf_matrix
