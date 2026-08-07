import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy import stats


class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.pca = None
        self.class_names = None
        self.feature_names_out = None

    def load_csv(self, file) -> pd.DataFrame:
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return pd.read_csv(file, encoding=enc)
            except (UnicodeDecodeError, Exception):
                try:
                    file.seek(0)
                except Exception:
                    pass
        raise ValueError("Could not decode CSV with common encodings.")

    def get_statistics(self, df: pd.DataFrame, feature_cols: list, label_col: str) -> dict:
        return {
            'shape': df.shape,
            'missing': df[feature_cols + [label_col]].isnull().sum(),
            'missing_pct': (df[feature_cols + [label_col]].isnull().sum() / len(df) * 100).round(2),
            'duplicates': df.duplicated().sum(),
            'class_counts': df[label_col].value_counts() if label_col else None,
            'description': df[feature_cols].describe() if feature_cols else None,
        }

    def clean(self, df: pd.DataFrame, feature_cols: list, label_col: str, config: dict) -> pd.DataFrame:
        df = df[feature_cols + [label_col]].copy()

        if config.get('remove_duplicates', True):
            df = df.drop_duplicates()

        strategy = config.get('missing_strategy', 'drop')
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
            df = df.dropna(subset=[label_col])
        elif strategy == 'median':
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
            df = df.dropna(subset=[label_col])
        elif strategy == 'mode':
            for col in feature_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)
            df = df.dropna(subset=[label_col])

        outlier_method = config.get('outlier_method', 'none')
        if outlier_method == 'iqr':
            q1 = df[feature_cols].quantile(0.25)
            q3 = df[feature_cols].quantile(0.75)
            iqr = q3 - q1
            mask = ~((df[feature_cols] < (q1 - 1.5 * iqr)) | (df[feature_cols] > (q3 + 1.5 * iqr))).any(axis=1)
            df = df[mask]
        elif outlier_method == 'zscore':
            threshold = config.get('zscore_threshold', 3.0)
            z = np.abs(stats.zscore(df[feature_cols].select_dtypes(include=np.number), nan_policy='omit'))
            mask = (z < threshold).all(axis=1)
            df = df[mask]

        return df.reset_index(drop=True)

    def pca_analysis(self, df: pd.DataFrame, feature_cols: list, normalization: str = 'standard') -> dict:
        """Fit full PCA on normalised data and return analysis info for visualisation."""
        X = df[feature_cols].dropna().values.astype(np.float32)
        scaler_map = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
        cls = scaler_map.get(normalization)
        if cls:
            X = cls().fit_transform(X)

        n_components = min(len(feature_cols), X.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(X)

        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_,          # (n_components, n_features)
            'feature_names': list(feature_cols),
            'n_components_total': n_components,
        }

    def preprocess(self, df: pd.DataFrame, feature_cols: list, label_col: str, config: dict) -> dict:
        X = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].astype(str).values   # coerce to str so encoder always produces string class names
        y = self.label_encoder.fit_transform(y_raw)
        self.class_names = self.label_encoder.classes_

        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        stratify = y if config.get('stratify', True) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        norm = config.get('normalization', 'standard')
        if norm == 'standard':
            self.scaler = StandardScaler()
        elif norm == 'minmax':
            self.scaler = MinMaxScaler()
        elif norm == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        feature_names = list(feature_cols)

        if config.get('use_pca', False):
            n_components = config.get('pca_components', 4)
            n_components = min(n_components, X_train.shape[1], X_train.shape[0] - 1)
            self.pca = PCA(n_components=n_components)
            X_train = self.pca.fit_transform(X_train)
            X_test = self.pca.transform(X_test)
            feature_names = [f'PC{i+1}' for i in range(n_components)]
            explained = self.pca.explained_variance_ratio_.sum()
        else:
            self.pca = None
            explained = None

        # Class balancing (train only)
        balancing = config.get('balancing', 'none')
        if balancing == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=random_state, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass
        elif balancing == 'undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=random_state)
                X_train, y_train = rus.fit_resample(X_train, y_train)
            except Exception:
                pass

        self.feature_names_out = feature_names

        return {
            'X_train': X_train.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_train': y_train,
            'y_test': y_test,
            'class_names': [str(c) for c in self.class_names],
            'n_features': X_train.shape[1],
            'n_classes': len(self.class_names),
            'feature_names': feature_names,
            'pca_explained_variance': explained,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
