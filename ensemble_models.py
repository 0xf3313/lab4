"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=4,
        random_state=data.random_seed,
        n_jobs=1,
    )
    train_target = np.asarray(data.train_target).ravel()
    test_target = np.asarray(data.test_target).ravel()
    model.fit(data.train_features, train_target)

    positive_class = list(model.classes_).index(1)
    probabilities = model.predict_proba(data.test_features)[:, positive_class]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Random Forest (150 деревьев, min_samples_leaf=4)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, zero_division=0)),
        recall=float(recall_score(test_target, predictions, zero_division=0)),
        f1=float(f1_score(test_target, predictions, zero_division=0)),
    )
