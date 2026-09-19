"""
Model Training & Evaluation Script for CareerSense AI.
Loads student profiles dataset, trains K-NN (primary) and Decision Tree (baseline),
computes classification metrics, confusion matrices, and feature importances,
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
    print("=" * 60)
    print(" CareerSense AI - Model Training & Evaluation Pipeline ")
    print("=" * 60)
    
    # Check if dataset exists, otherwise generate
    if not DATASET_PATH.exists():
        print("Dataset not found. Generating realistic student cohort dataset...")
        generate_student_dataset()

    print("Initializing CareerClassifier...")
    classifier = CareerClassifier()
    
    print("Training K-NN (Primary) and Decision Tree (Baseline) models...")
    metrics = classifier.train_models()
    
    knn = metrics["knn"]
    dt = metrics["decision_tree"]
    info = metrics["dataset_info"]
    
    print(f"\nDataset Overview:")
    print(f"- Total Samples: {info['total_samples']}")
    print(f"- Training Set: {info['training_samples']} (80%)")
    print(f"- Testing Set: {info['testing_samples']} (20%)")
    print(f"- Feature Vector Dimension: {info['features_count']}")
    
    print("\n" + "-" * 50)
    print(f"1. {knn['name']} Results:")
    print(f"   Accuracy : {knn['accuracy']}%")
    print(f"   Precision: {knn['precision']}%")
    print(f"   Recall   : {knn['recall']}%")
    print(f"   F1-Score : {knn['f1_score']}%")
    print("   Confusion Matrix:")
    for row in knn['confusion_matrix']:
        print(f"     {row}")
        
    print("\n" + "-" * 50)
    print(f"2. {dt['name']} Results:")
    print(f"   Accuracy : {dt['accuracy']}%")
    print(f"   Precision: {dt['precision']}%")
    print(f"   Recall   : {dt['recall']}%")
    print(f"   F1-Score : {dt['f1_score']}%")
    print("   Top Influential Features in Decision Tree:")
    for feat, imp in dt['top_features']:
        print(f"     * {feat}: {imp}")
        
    print("\n" + "=" * 60)
    print("All models successfully trained and persisted to ai_engine/saved_models/")
    print("=" * 60)

if __name__ == "__main__":
    main()
