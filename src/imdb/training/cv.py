import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from typing import Callable, Any, Tuple
import structlog
import mlflow
from tqdm import tqdm

logger = structlog.get_logger(__name__)

class TqdmOptunaCallback:
    def __init__(self, n_trials: int, desc: str) -> None:
        self.pbar = tqdm(total=n_trials, desc=desc, leave=False)
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.pbar.update(1)
    def close(self) -> None:
        self.pbar.close()

def run_nested_cv(
    X: Any, 
    y: Any, 
    objective_fn: Callable,
    fit_predict_fn: Callable,
    n_splits: int = 5,
    n_trials: int = 20,
    random_state: int = 42,
    direction: str = "maximize",
    mlflow_run_name: str = "Nested_CV"
) -> Tuple[np.ndarray, dict, list]:
    """
    Universal Nested Cross-Validation loop for any model type.
    """
    n_samples = len(y)
    oof_preds = np.full(n_samples, np.nan, dtype=np.float64)
    outer_metrics = []
    best_params_across_folds = []
    
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    with mlflow.start_run(run_name=mlflow_run_name):
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(n_samples), y)):
            logger.info("outer_fold_start", fold=fold_idx + 1, total=n_splits)
            
            # Sub-slicing logic depending on if X is a tuple/list of inputs (e.g., text+meta separately)
            # or a single numpy/tensor array
            if isinstance(X, tuple):
                X_train = tuple(x[train_idx] for x in X)
                X_test = tuple(x[test_idx] for x in X)
            else:
                X_train, X_test = X[train_idx], X[test_idx]
                
            y_train, y_test = y[train_idx], y[test_idx]
            
            # --- Inner Loop: HPO ---
            study = optuna.create_study(direction=direction)
            cb = TqdmOptunaCallback(n_trials, desc=f"Fold {fold_idx + 1} Optuna")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            study.optimize(
                lambda trial: objective_fn(trial, X_train, y_train),
                n_trials=n_trials,
                callbacks=[cb]
            )
            cb.close()
            
            best_params = study.best_params
            best_params_across_folds.append((study.best_value, best_params))
            
            # --- Outer Loop: Refit and Predict ---
            preds, outer_metric = fit_predict_fn(X_train, y_train, X_test, y_test, best_params)
            oof_preds[test_idx] = preds
            outer_metrics.append(outer_metric)
            
            logger.info("outer_fold_done", fold=fold_idx + 1, metric=round(outer_metric, 4))
            mlflow.log_metric(f"fold_{fold_idx + 1}_metric", outer_metric, step=fold_idx)
            
        nested_metric = float(np.mean(outer_metrics))
        nested_std = float(np.std(outer_metrics))
        logger.info("nested_cv_done", metric=round(nested_metric, 4), std=round(nested_std, 4))
        
        mlflow.log_metric("true_nested_cv_metric", nested_metric)
        mlflow.log_metric("nested_cv_std", nested_std)
        
        # Log Best Overall
        best_overall_score, best_overall_params = max(best_params_across_folds, key=lambda x: x[0])
        mlflow.log_params({f"best_overall_{k}": v for k, v in best_overall_params.items()})
        
    return oof_preds, best_overall_params, outer_metrics
