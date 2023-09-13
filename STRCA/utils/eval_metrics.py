import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

def fault_type_classification(y_true, y_pred):

    evaluation_metric_dict = dict()

    evaluation_metric_dict['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    for i in ['micro', 'macro']:
        for j in [precision_score, recall_score, f1_score]:
            evaluation_metric_dict[f'{i}_{j.__name__}'] = j(y_true, y_pred, average=i, zero_division=0)

    for j in [precision_score, recall_score, f1_score]:
        evaluation_metric_dict[f'{j.__name__}'] = j(y_true, y_pred, average=None, zero_division=0)

    return evaluation_metric_dict


def evaluate(y_true, y_pred):
    fc_result = fault_type_classification(y_true, y_pred)
    logger.info(f"confusion matrix:\n{fc_result['confusion_matrix']}")
    convert = {
        'p': 'precision',
        'r': 'recall',
        'f1': 'f1'
    }
    for em in ['p', 'r', 'f1']:
        logger.info(
            f'{convert[em].ljust(9)} | micro: {fc_result["micro_" + convert[em] + "_score"]:.6f}; macro: {fc_result["macro_" + convert[em] + "_score"]:.6f}; score: {fc_result[convert[em] + "_score"]}')
