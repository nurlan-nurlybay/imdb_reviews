import os
from typing import Any, List
import matplotlib.pyplot as plt
import shap
import structlog
from catboost import CatBoostClassifier

logger = structlog.get_logger(__name__)

def generate_shap_plot(
    model: CatBoostClassifier, 
    X: Any, 
    X_raw: Any, 
    feature_names: List[str], 
    output_path: str, 
    title: str,
    max_display: int = 20
) -> None:
    """
    Generates a SHAP beeswarm plot for CatBoost models.
    """
    logger.info("generating_shap", title=title, num_features=len(feature_names))
    
    explainer = shap.TreeExplainer(model)
    # Use the Pool (X) for calculation
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    
    # If it's a CSR matrix, convert to dense for plotting
    if hasattr(X_raw, "toarray"):
        X_plot = X_raw.toarray()
    else:
        X_plot = X_raw
        
    shap.summary_plot(
        shap_values, 
        X_plot, 
        feature_names=feature_names, 
        max_display=max_display, 
        show=False
    )
    
    plt.title(title, pad=20, fontsize=16)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("shap_plot_saved", path=output_path)
