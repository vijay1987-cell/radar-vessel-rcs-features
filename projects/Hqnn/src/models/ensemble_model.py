import numpy as np
import torch


class EnsembleModel:
    """
    Combines predictions from multiple trained models via soft or hard voting.

    models_meta: list of dicts, each with keys:
        'model'    — the trained model object
        'is_torch' — bool
        'name'     — display name
    """

    def __init__(self, models_meta: list, voting: str = 'soft'):
        if len(models_meta) < 2:
            raise ValueError("Ensemble requires at least 2 models.")
        self.models_meta = models_meta
        self.voting = voting

    def _proba(self, meta: dict, X: np.ndarray) -> np.ndarray:
        # If model was loaded from disk the object is None — use stored test probabilities
        if meta.get('probabilities') is not None and meta.get('model') is None:
            return meta['probabilities']

        model = meta['model']
        if model is None:
            raise ValueError(
                f"Model '{meta['name']}' has no stored probabilities and no model object. "
                "Re-train it in the current session."
            )

        if meta['is_torch']:
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32))
                return torch.softmax(logits, dim=1).numpy()

        # sklearn-style (ClassicalModel wrapper or QKEClassifier)
        inner = getattr(model, 'model', model)
        if hasattr(inner, 'predict_proba'):
            return inner.predict_proba(X)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        preds = model.predict(X)
        n = int(preds.max()) + 1
        oh = np.zeros((len(preds), n))
        oh[np.arange(len(preds)), preds] = 1.0
        return oh

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [self._proba(m, X) for m in self.models_meta]
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voting == 'soft':
            return self.predict_proba(X).argmax(axis=1)
        # hard voting: majority vote per sample
        votes = np.array([self._proba(m, X).argmax(axis=1) for m in self.models_meta])
        result = np.zeros(votes.shape[1], dtype=int)
        for i in range(votes.shape[1]):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            result[i] = vals[counts.argmax()]
        return result
