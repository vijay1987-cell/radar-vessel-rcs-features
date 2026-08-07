import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class ClassicalModel:
    """Wrapper for scikit-learn classifiers with a unified interface."""

    MODELS = {
        'Random Forest': RandomForestClassifier,
        'SVM': SVC,
        'MLP': MLPClassifier,
        'Gradient Boosting': GradientBoostingClassifier,
    }
    if _XGB_AVAILABLE:
        MODELS['XGBoost'] = _XGBClassifier

    def __init__(self, model_type: str, params: dict):
        self.model_type = model_type
        self.params = params
        cls = self.MODELS.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model = cls(**params)
        self.history = {'train_acc': [], 'val_acc': []}

    def fit(self, X_train, y_train, X_val=None, y_val=None, callback=None):
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        val_acc = accuracy_score(y_val, self.model.predict(X_val)) if X_val is not None else None
        self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if callback:
            callback({'epoch': 1, 'train_loss': 0.0, 'train_acc': train_acc, 'val_acc': val_acc})

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        decision = self.model.decision_function(X)
        # softmax for SVM decision function
        exp = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
