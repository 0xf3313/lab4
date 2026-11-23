import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


def metrics(target, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def soft_voting(seed, depth, weights):
    forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=4, random_state=seed, n_jobs=1)
    boosting = GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05, max_depth=depth,
        min_samples_leaf=10, random_state=seed,
    )
    return VotingClassifier(
        estimators=[("forest", forest), ("boosting", boosting)],
        voting="soft", weights=weights, n_jobs=1,
    )


def main():
    folder = Path(__file__).resolve().parent
    seed = 42
    data = pd.read_csv(folder / "german.csv", sep=";")
    features = data.drop(columns="Creditability")
    target = data["Creditability"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=seed, stratify=target
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=4, random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=4, random_state=seed, n_jobs=1),
        "Gradient Boosting depth 1": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=1, min_samples_leaf=10, random_state=seed),
        "Gradient Boosting depth 2": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, min_samples_leaf=10, random_state=seed),
        "Gradient Boosting depth 3": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, min_samples_leaf=10, random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=20, random_state=seed),
    }
    voting_models = {
        "Soft Voting RF + GB2 (1:1)": soft_voting(seed, depth=2, weights=[1, 1]),
        "Soft Voting RF + GB3 (1:1)": soft_voting(seed, depth=3, weights=[1, 1]),
        "Soft Voting RF + GB2 (2:1)": soft_voting(seed, depth=2, weights=[2, 1]),
        "Soft Voting RF + GB3 (2:1)": soft_voting(seed, depth=3, weights=[2, 1]),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    validation = {}
    for name, model in {**models, **voting_models}.items():
        scores = cross_val_score(model, x_train, y_train, scoring="roc_auc", cv=folds, n_jobs=1)
        validation[name] = {"mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std())}

    chosen_name = max(validation, key=lambda name: validation[name]["mean_roc_auc"])
    if chosen_name != "Random Forest":
        raise RuntimeError("По CV выбран другой вариант; проверьте настройки контрактной функции.")
    best_voting = max(voting_models, key=lambda name: validation[name]["mean_roc_auc"])

    # Выбор завершён до расчёта метрик на отложенной выборке.
    test_results = {}
    test_models = {**models, best_voting: voting_models[best_voting]}
    for name, model in test_models.items():
        model.fit(x_train, y_train)
        positive_class = list(model.classes_).index(1)
        probabilities = model.predict_proba(x_test)[:, positive_class]
        test_results[name] = metrics(y_test, probabilities)

    result = run_ensemble_experiment(EnsembleInput(x_train, y_train, x_test, y_test, random_seed=seed))
    selected_metrics = {key: getattr(result, key) for key in ("roc_auc", "accuracy", "precision", "recall", "f1")}
    if not all(np.isclose(selected_metrics[key], test_results[chosen_name][key]) for key in selected_metrics):
        raise RuntimeError("Метрики контрактной функции отличаются от эксперимента.")

    output = {
        "model_name": result.model_name,
        "random_seed": seed,
        "test_size": 0.2,
        "selection_criterion": "mean ROC-AUC, stratified 5-fold CV on training data",
        "selected_model": chosen_name,
        "best_voting_model": best_voting,
        "metrics": selected_metrics,
        "validation": validation,
        "test_results": test_results,
    }
    (folder / "experiment_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Число пропусков: {int(data.isna().sum().sum())}")


if __name__ == "__main__":
    main()
