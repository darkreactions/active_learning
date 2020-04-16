from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier


def get_learner(X_training, y_training):
    return ActiveLearner(estimator=RandomForestClassifier(),
                         X_training=X_training, y_training=y_training)
