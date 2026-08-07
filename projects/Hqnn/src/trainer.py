import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize


class TorchTrainer:
    """Trains PyTorch-based models (VQC, HQNN)."""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
        }

        opt_map = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
        }
        opt_cls = opt_map.get(config.get('optimizer', 'adam'), torch.optim.Adam)
        lr = config.get('learning_rate', 1e-3)
        wd = config.get('weight_decay', 0.0)
        self.optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)
        class_weights = config.get('class_weights')
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train, y_train, X_val, y_val, callback=None, patience=None):
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 30)
        if patience is None:
            patience = self.config.get('patience', None)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.long)
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.long)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

        best_val_acc = -1.0
        best_state = None
        patience_counter = 0
        self.best_epoch = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            ep_loss, ep_correct = 0.0, 0

            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                ep_loss += loss.item() * len(xb)
                ep_correct += (logits.argmax(1) == yb).sum().item()

            train_loss = ep_loss / len(X_train)
            train_acc = ep_correct / len(X_train)

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_v)
                val_loss = self.criterion(val_logits, y_v).item()
                val_acc = (val_logits.argmax(1) == y_v).float().mean().item()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # best-checkpoint tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if callback:
                stop = callback({
                    'epoch': epoch, 'total_epochs': epochs,
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc,
                    'best_val_acc': best_val_acc, 'best_epoch': self.best_epoch,
                })
                if stop:
                    break

            if patience and patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} "
                      f"(best val_acc {best_val_acc:.1%} at epoch {self.best_epoch})")
                break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history


def evaluate_model(model, X_test, y_test, class_names: list, is_torch: bool = True):
    """Compute full evaluation metrics for a trained model."""
    if is_torch:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_t)
            proba = torch.softmax(logits, dim=1).numpy()
            preds = logits.argmax(1).numpy()
    else:
        preds = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=class_names,
                                   output_dict=True, zero_division=0)

    auc = None
    if proba is not None and len(class_names) > 1:
        try:
            y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            auc = roc_auc_score(y_bin, proba, average='macro', multi_class='ovr')
        except Exception:
            auc = None

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': preds,
        'probabilities': proba,
    }
