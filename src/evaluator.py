"""
Model evaluation and comparative analysis.

Functions:
  - evaluate_model     : compute and print all required metrics
  - plot_confusion_matrix : save a confusion-matrix heatmap
  - compare_results    : build a summary DataFrame and save a bar chart
  - print_misclassified: show example errors for qualitative analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    label: str = "Model",
    save_dir: str = "outputs",
) -> dict:
    """Return a metrics dict and print a full classification report."""
    os.makedirs(save_dir, exist_ok=True)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    plot_confusion_matrix(y_test, y_pred, label=label, save_dir=save_dir)

    return {"label": label, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, label: str, save_dir: str = "outputs") -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    safe_name = label.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(save_dir, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix: {path}")


def compare_results(results: list[dict], save_dir: str = "outputs") -> pd.DataFrame:
    """
    Build a summary table and a grouped bar chart from a list of
    metric dicts returned by evaluate_model().
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results).set_index("label")
    metrics = ["accuracy", "precision", "recall", "f1"]

    print("\n" + "="*55)
    print("  COMPARATIVE SUMMARY")
    print("="*55)
    print(df[metrics].to_string(float_format="{:.4f}".format))

    # Bar chart
    ax = df[metrics].plot(kind="bar", figsize=(12, 5), colormap="tab10", edgecolor="black")
    ax.set_title("Model Comparison — All Configurations")
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n  Saved comparison chart: {path}")

    return df


def print_misclassified(
    model,
    X_test,
    y_test,
    original_texts: list[str],
    n: int = 10,
    label: str = "Model",
) -> None:
    """Print `n` misclassified examples to help understand error patterns."""
    y_pred = model.predict(X_test)
    misclassified_idx = np.where(y_pred != np.array(y_test))[0]

    print(f"\n  Misclassified examples ({label}) — showing {min(n, len(misclassified_idx))}:")
    for i in misclassified_idx[:n]:
        true_label = "Positive" if y_test.iloc[i] == 1 else "Negative"
        pred_label = "Positive" if y_pred[i] == 1 else "Negative"
        snippet = original_texts[i][:120].replace("\n", " ")
        print(f"    [{i}] True={true_label}, Pred={pred_label}")
        print(f"         \"{snippet}\"")
