import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def plot_pca_scree(analysis: dict):
    """Scree plot: per-component and cumulative explained variance."""
    evr = analysis['explained_variance_ratio']
    cum = analysis['cumulative_variance']
    idx = list(range(1, len(evr) + 1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=idx, y=(evr * 100).tolist(), name='Per component (%)',
               marker_color='steelblue'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=idx, y=(cum * 100).tolist(), name='Cumulative (%)',
                   line=dict(color='firebrick', width=2), mode='lines+markers'),
        secondary_y=True,
    )
    fig.add_hline(y=90, line_dash='dash', line_color='grey',
                  annotation_text='90 %', secondary_y=True)
    fig.update_layout(
        title='PCA Scree Plot — Explained Variance',
        xaxis_title='Principal Component',
        yaxis_title='Variance explained (%)',
        yaxis2_title='Cumulative variance (%)',
        height=380,
    )
    return fig


def plot_pca_loadings(analysis: dict, n_show: int = 4):
    """Heatmap of feature loadings for the top N principal components."""
    n_show = min(n_show, analysis['n_components_total'])
    components = analysis['components'][:n_show]      # (n_show, n_features)
    feature_names = analysis['feature_names']
    pc_labels = [f'PC{i+1}' for i in range(n_show)]

    fig = go.Figure(data=go.Heatmap(
        z=components.tolist(),
        x=feature_names,
        y=pc_labels,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Loading'),
        text=[[f'{v:.3f}' for v in row] for row in components],
        texttemplate='%{text}',
    ))
    fig.update_layout(
        title=f'Feature Loadings — Top {n_show} Principal Components',
        xaxis_title='Original Feature',
        yaxis_title='Principal Component',
        xaxis=dict(tickangle=-40),
        height=max(300, 60 * n_show + 150),
    )
    return fig


def plot_pca_variance_bar(analysis: dict, n_components: int):
    """Bar chart showing how much variance is captured by the chosen n_components."""
    cum = float(analysis['cumulative_variance'][n_components - 1]) * 100
    remaining = 100 - cum
    fig = go.Figure(go.Bar(
        x=[f'PC 1–{n_components} (selected)', 'Discarded'],
        y=[cum, remaining],
        marker_color=['seagreen', 'lightcoral'],
        text=[f'{cum:.1f}%', f'{remaining:.1f}%'],
        textposition='auto',
    ))
    fig.update_layout(
        title=f'Variance Retained with {n_components} Components',
        yaxis=dict(range=[0, 100], title='Variance (%)'),
        height=300,
    )
    return fig


def plot_class_distribution(class_counts, title="Class Distribution"):
    fig = px.bar(
        x=class_counts.index.tolist(),
        y=class_counts.values.tolist(),
        labels={'x': 'Class', 'y': 'Count'},
        title=title,
        color=class_counts.values.tolist(),
        color_continuous_scale='Blues',
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def plot_training_curves(history: dict, model_name: str = ""):
    has_loss = 'train_loss' in history and len(history['train_loss']) > 0
    has_acc = 'train_acc' in history and len(history['train_acc']) > 0

    rows = sum([has_loss, has_acc])
    if rows == 0:
        return None

    fig = make_subplots(rows=rows, cols=1, subplot_titles=(
        (['Loss'] if has_loss else []) + (['Accuracy'] if has_acc else [])
    ))

    row = 1
    epochs = list(range(1, len(history.get('train_loss', history.get('train_acc'))) + 1))

    if has_loss:
        fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss',
                                  line=dict(color='steelblue')), row=row, col=1)
        if 'val_loss' in history and history['val_loss']:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss',
                                      line=dict(color='orange', dash='dash')), row=row, col=1)
        row += 1

    if has_acc:
        fig.add_trace(go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc',
                                  line=dict(color='green')), row=row, col=1)
        if 'val_acc' in history and history['val_acc']:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc',
                                      line=dict(color='red', dash='dash')), row=row, col=1)

    fig.update_layout(title=f"Training Curves — {model_name}", height=400 * rows)
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str = ""):
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [[f"{cm[i,j]}<br>({cm_normalized[i,j]:.1%})" for j in range(len(class_names))]
            for i in range(len(class_names))]

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        text=text,
        texttemplate="%{text}",
        colorscale='Blues',
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange='reversed'),
    )
    return fig


def plot_roc_curves(y_test, probabilities, class_names: list, model_name: str = ""):
    if probabilities is None:
        return None
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig = go.Figure()
    for i, name in enumerate(class_names):
        try:
            if i < probabilities.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.2f})'))
        except Exception:
            pass

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='grey')))
    fig.update_layout(
        title=f"ROC Curves (OvR) — {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=450,
    )
    return fig


def plot_model_comparison(trained_models: dict):
    """Bar chart comparing accuracy, F1-macro, and AUC across trained models."""
    names, accs, f1s, aucs = [], [], [], []
    for key, m in trained_models.items():
        metrics = m.get('metrics', {})
        names.append(key)
        accs.append(round(metrics.get('accuracy', 0) * 100, 2))
        f1s.append(round(metrics.get('f1_macro', 0) * 100, 2))
        auc_val = metrics.get('auc_macro')
        aucs.append(round(auc_val * 100, 2) if auc_val is not None else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy (%)', x=names, y=accs, marker_color='steelblue'))
    fig.add_trace(go.Bar(name='F1 Macro (%)', x=names, y=f1s, marker_color='seagreen'))
    fig.add_trace(go.Bar(name='AUC Macro (%)', x=names, y=aucs, marker_color='coral'))
    fig.update_layout(
        barmode='group',
        title='Model Comparison',
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
    )
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list, model_name: str = ""):
    idx = np.argsort(importances)[::-1]
    fig = px.bar(
        x=[feature_names[i] for i in idx],
        y=[importances[i] for i in idx],
        title=f"Feature Importance — {model_name}",
        labels={'x': 'Feature', 'y': 'Importance'},
        color=[importances[i] for i in idx],
        color_continuous_scale='Viridis',
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig
