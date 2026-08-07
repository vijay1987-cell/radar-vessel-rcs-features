import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class QKEClassifier:
    """
    Quantum Kernel Estimation Classifier.
    Computes K(xi, xj) = |<φ(xi)|φ(xj)>|² using a quantum feature map,
    then passes the kernel matrix to a classical SVM.

    NOTE: scales as O(N²) — automatically subsamples if N > max_train_samples.
    """

    def __init__(self, n_qubits: int, n_layers: int, embedding: str = 'zzfeaturemap',
                 svm_c: float = 1.0, max_train_samples: int = 200):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.embedding = embedding
        self.svm_c = svm_c
        self.max_train_samples = max_train_samples
        self.svm = SVC(kernel='precomputed', C=svm_c, probability=True, class_weight='balanced')
        self.X_train_sub = None
        self.history = {'train_acc': [], 'val_acc': []}

        try:
            dev = qml.device('lightning.qubit', wires=n_qubits)
        except Exception:
            dev = qml.device('default.qubit', wires=n_qubits)

        @qml.qnode(dev, interface='numpy')
        def _feature_map(x):
            if embedding == 'angle':
                qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
                for _ in range(n_layers):
                    qml.BasicEntanglerLayers(
                        np.zeros((1, n_qubits)), wires=range(n_qubits)
                    )
            elif embedding == 'zzfeaturemap':
                for rep in range(n_layers):
                    for i in range(n_qubits):
                        qml.Hadamard(wires=i)
                        qml.RZ(2.0 * x[i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
                        qml.CNOT(wires=[i, i + 1])
            return qml.state()

        self._feature_map = _feature_map

    def _kernel(self, A, B):
        """Compute kernel matrix |<φ(a)|φ(b)>|² for all pairs."""
        K = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                sa = self._feature_map(a)
                sb = self._feature_map(b)
                K[i, j] = np.abs(np.dot(np.conj(sa), sb)) ** 2
        return K

    def _project(self, X):
        """Reduce to n_qubits features using linear projection if needed."""
        n = self.n_qubits
        if X.shape[1] > n:
            X = X[:, :n]  # take first n features (PCA should have been applied upstream)
        elif X.shape[1] < n:
            pad = np.zeros((X.shape[0], n - X.shape[1]))
            X = np.hstack([X, pad])
        return X

    def fit(self, X_train, y_train, X_val=None, y_val=None, callback=None):
        X_train = self._project(X_train)

        # Subsample if too large
        if len(X_train) > self.max_train_samples:
            idx = np.random.choice(len(X_train), self.max_train_samples, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        self.X_train_sub = X_train
        self.y_train_sub = y_train

        K_train = self._kernel(X_train, X_train)
        self.svm.fit(K_train, y_train)

        train_acc = accuracy_score(y_train, self.svm.predict(K_train))
        self.history['train_acc'].append(train_acc)

        val_acc = None
        if X_val is not None:
            X_val = self._project(X_val)
            K_val = self._kernel(X_val, self.X_train_sub)
            val_acc = accuracy_score(y_val, self.svm.predict(K_val))
            self.history['val_acc'].append(val_acc)

        if callback:
            callback({'epoch': 1, 'train_loss': 0.0, 'train_acc': train_acc, 'val_acc': val_acc})

    def predict(self, X):
        X = self._project(X)
        K = self._kernel(X, self.X_train_sub)
        return self.svm.predict(K)

    def predict_proba(self, X):
        X = self._project(X)
        K = self._kernel(X, self.X_train_sub)
        return self.svm.predict_proba(K)
