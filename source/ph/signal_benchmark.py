"""Topological signal benchmark evaluating persistent homology discriminability."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from source.data.candidate_extractor import CandidateExample
from source.ph.persistence_extractor import PersistenceExtractor
from source.ph.ph_graph_view import PHGraphView
from source.ph.vectorizer import PersistenceVectorizer


class TopologicalSignalBenchmark:
    """Benchmark evaluating discriminative power of persistent homology feature vectors."""

    def __init__(self, vectorizer: Optional[PersistenceVectorizer] = None, random_state: int = 42):
        self.vectorizer = vectorizer or PersistenceVectorizer()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, random_state=random_state)

    def extract_features_and_labels(
        self,
        candidates: Sequence[CandidateExample],
        filtration_type: str = "temporal",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vectorized topological embeddings and target labels for a candidate list."""
        x_list: List[np.ndarray] = []
        y_list: List[int] = []

        for cand in candidates:
            ph_view = PHGraphView.from_candidate_example(cand, filtration_type=filtration_type)
            diag = PersistenceExtractor.compute_diagram(ph_view, cap_infinity=True)
            vec = self.vectorizer.vectorize(diag)
            x_list.append(vec)
            y_list.append(int(cand.target_y))

        if x_list:
            X = np.array(x_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int64)
        else:
            X = np.zeros((0, self.vectorizer.output_dim), dtype=np.float32)
            y = np.zeros(0, dtype=np.int64)

        return X, y

    def fit_and_evaluate(
        self,
        train_candidates: Sequence[CandidateExample],
        val_candidates: Sequence[CandidateExample],
        filtration_type: str = "temporal",
    ) -> Dict[str, Any]:
        """Train logistic regression on train topological vectors and evaluate on validation set."""
        X_train, y_train = self.extract_features_and_labels(train_candidates, filtration_type=filtration_type)
        X_val, y_val = self.extract_features_and_labels(val_candidates, filtration_type=filtration_type)

        if len(np.unique(y_train)) < 2:
            raise ValueError("Training set must contain both positive (1) and negative (0) examples.")
        if len(np.unique(y_val)) < 2:
            raise ValueError("Validation set must contain both positive (1) and negative (0) examples.")

        # Standardize features with zero-variance safety
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Replace any residual NaNs from zero-variance columns with 0.0
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

        # Train linear model
        self.classifier.fit(X_train_scaled, y_train)

        # Predict probability of positive class (y=1)
        val_probs = self.classifier.predict_proba(X_val_scaled)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_val, val_probs))
        pr_auc = float(average_precision_score(y_val, val_probs))
        accuracy = float(np.mean(val_preds == y_val))

        return {
            "val_roc_auc": roc_auc,
            "val_pr_auc": pr_auc,
            "val_accuracy": accuracy,
            "feature_dim": int(self.vectorizer.output_dim),
            "num_train": int(len(train_candidates)),
            "num_val": int(len(val_candidates)),
            "filtration_type": filtration_type,
        }
