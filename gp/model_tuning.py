from __future__ import division
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


N_CLASSES = 3
CLASSES = ['Pass', 'Shot', 'Take on']
FEATURES = ["x_real", "y_real",'score']


def validate_model(models, eval_data, naive_probs):
    # evaluate model error on validation set and for naive baseline
    model_error = evaluate_error(models, eval_data)
    baseline_error = evaluate_baseline(models, eval_data, naive_probs)
    return model_error, baseline_error


def evaluate_baseline(models, eval_data, naive_probs):
    # evaluate baseline across all players
    # this is also dumb and should in one function
    error = []
    weights = []
    for id in models.keys():
        test = eval_data[eval_data.player_id == id]
        if (test.groupby('event_type').size().min() >= 2) and (test.event_type.nunique() == 3):
            micro_roc = score_baseline(models[id], eval_data, naive_probs)
            error.append(micro_roc)
            weights.append(test.shape[0])
    return np.average(error, weights=weights)

def score_baseline(model, eval_data, naive_probs):
    # evaluate baseline predictions
    X = eval_data[FEATURES]
    y = eval_data.event_type
    # match the empirical probs in right order
    probs = [naive_probs[x] for x in model.classes_]
    y_score = np.vstack([probs for x in range(len(y))])
    y_test = label_binarize(y, classes=CLASSES)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        if (len(CLASSES) == y_score.shape[1]):
            # this is dumb shouldnt recreate this
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            return .5
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def evaluate_error(models, eval_data):
    # evalute error for all players in test data
    error = []
    weights = []
    for id in models.keys():
        test = eval_data[eval_data.player_id == id]
        if (test.groupby('event_type').size().min() >= 2) and (test.event_type.nunique() == 3):
            # can't calc auc on 1 point
            micro_roc = score_model(models[id], test)
            error.append(micro_roc)
            weights.append(test.shape[0])
    # return error weighted by players total actions
    return np.average(error, weights=weights)

    
def score_model(model, eval_data):
    # score model and get error for a given model and player data set
    X = eval_data[FEATURES]
    y = eval_data.event_type
    
    y_score = model.predict_proba(X)
    y_test = label_binarize(y, classes=CLASSES)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        if (len(CLASSES) == y_score.shape[1]):
            # model was trained on fewer actions than it was tested on
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            return .5
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def split_data(data):
    # split data according to train-test-validate flags
    train = data[data.train == 1].reset_index()
    test = data[data.test == 1].reset_index()
    validate = data[data.validate == 1].reset_index()
    return train, test, validate


def build_models(data, kernel):
     # build a GP per player, return in dict keyed on id
    models = {}
    for id in data.player_id.unique():
        player_training = data[data.player_id == id]
        X = player_training[FEATURES]
        y = player_training.event_type
        gp_opt = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=2)
        try:
            gp_opt.fit(X, y)
            models[id] = gp_opt
        except ValueError as e:
            pass
    return models
