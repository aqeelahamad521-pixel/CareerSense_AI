"""
Model Training & Evaluation Script for CareerSense AI.
Loads student profiles dataset, trains K-NN (primary) and Decision Tree (baseline),
benchmarks against Dummy, Logistic Regression, and Random Forest classifiers,
computes cross-validation benchmarks, confusion matrices, and feature importances,
and saves the serialized model artifacts.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config import DATASET_PATH
from scripts.generate_dataset import generate_student_dataset
from ai_engine.ml_classifier import CareerClassifier

def main():
    print("=" * 70)
    print(" CareerSense AI - Machine Learning Model Training & Evaluation Pipeline ")
    print("=" * 70)
    
    # Check if dataset exists, otherwise generate
    if not DATASET_PATH.exists():
        print("Dataset not found. Generating realistic student cohort dataset...")
        generate_student_dataset()

    print("Initializing CareerClassifier...")
    classifier = CareerClassifier()
    
    print("Training models with 5-Fold Stratified Cross-Validation & held-out test split...")
    metrics = classifier.train_models()
    
    knn = metrics["knn"]
    dt = metrics["decision_tree"]
    info = metrics["dataset_info"]
    comp = metrics.get("model_comparison", [])
    
    print(f"\nDataset Overview:")
    print(f"- Total Samples: {info['total_samples']}")
    print(f"- Training Set: {info['training_samples']} (80%)")
    print(f"- Testing Set: {info['testing_samples']} (20%)")
    print(f"- Feature Vector Dimension: {info['features_count']}")
    print(f"- Class Distribution: {info.get('class_distribution', {})}")
    
    print("\n" + "=" * 70)
    print(" MODEL BENCHMARK COMPARISON TABLE (HELD-OUT TEST SET, N=170)")
    print("=" * 70)
    print(f"{'Model':<35} | {'Role':<22} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 95)
    for m in comp:
        print(f"{m['model']:<35} | {m.get('type', m.get('role', '')): <22} | {m['accuracy']:>8.2f}% | {m['f1_macro']:>8.2f}% | {m['f1_weighted']:>8.2f}%")

    print("\n" + "-" * 70)
    print(f"1. {knn['name']} (Selected Primary Model)")
    print(f"   5-Fold Cross-Validation Accuracy: {knn.get('cv_accuracy_mean', 'N/A')}% (+/- {knn.get('cv_accuracy_std', 'N/A')}%)")
    print(f"   Held-Out Test Accuracy          : {knn['accuracy']}%")
    print(f"   Weighted Precision / Recall / F1: {knn['precision']}% / {knn['recall']}% / {knn['f1_score']}%")
    print(f"   Macro Precision / Recall / F1   : {knn.get('precision_macro', 'N/A')}% / {knn.get('recall_macro', 'N/A')}% / {knn.get('f1_score_macro', 'N/A')}%")
    print("\n   Confusion Matrix (Rows=Actual, Columns=Predicted):")
    labels = knn.get("labels", [])
    print(f"   {'':<22} " + " ".join(f"{l[:7]:>8}" for l in labels))
    for i, row in enumerate(knn['confusion_matrix']):
        row_str = " ".join(f"{val:>8}" for val in row)
        print(f"   {labels[i]:<22} {row_str}")

    print("\n   Per-Class Performance (K-NN):")
    for trk, stats in knn.get("per_class", {}).items():
        print(f"   * {trk:<22}: Precision={stats['precision']:>5.1f}%, Recall={stats['recall']:>5.1f}%, F1={stats['f1_score']:>5.1f}%, Support={stats['support']}")

    print("\n" + "-" * 70)
    print(f"2. {dt['name']} (Interpretable Baseline Model)")
    print(f"   5-Fold Cross-Validation Accuracy: {dt.get('cv_accuracy_mean', 'N/A')}% (+/- {dt.get('cv_accuracy_std', 'N/A')}%)")
    print(f"   Held-Out Test Accuracy          : {dt['accuracy']}%")
    print(f"   Weighted Precision / Recall / F1: {dt['precision']}% / {dt['recall']}% / {dt['f1_score']}%")
    print(f"   Macro Precision / Recall / F1   : {dt.get('precision_macro', 'N/A')}% / {dt.get('recall_macro', 'N/A')}% / {dt.get('f1_score_macro', 'N/A')}%")
    print("\n   Confusion Matrix (Rows=Actual, Columns=Predicted):")
    print(f"   {'':<22} " + " ".join(f"{l[:7]:>8}" for l in labels))
    for i, row in enumerate(dt['confusion_matrix']):
        row_str = " ".join(f"{val:>8}" for val in row)
        print(f"   {labels[i]:<22} {row_str}")

    print("\n   Top Influential Gini Features in Decision Tree:")
    for feat, imp in dt.get('top_features', []):
        print(f"     * {feat:<25}: {imp:.4f}")
        
    print("\n" + "=" * 70)
    print("All models successfully trained, evaluated, and persisted.")
    print("Evaluation report saved to ai_engine/saved_models/evaluation_metrics.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
