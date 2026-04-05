"""
Train the spam detection model and save artifacts.
Run:  python train_model.py
"""

import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from spam_detector import load_data, train_model, save_model


def main():
    print("=" * 60)
    print("  Spam Email Detection - Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading dataset...")
    df = load_data()
    print(f"    Total messages : {len(df)}")
    print(f"    Ham messages   : {(df['label'] == 'ham').sum()}")
    print(f"    Spam messages  : {(df['label'] == 'spam').sum()}")

    # 2. Train model
    print("\n[2] Training model...")
    model, vectorizer, metrics = train_model(df)

    # 3. Print results
    print("\n[3] Evaluation Metrics:")
    print(f"    Accuracy  : {metrics['accuracy']}%")
    print(f"    Precision : {metrics['precision']}%")
    print(f"    Recall    : {metrics['recall']}%")
    print(f"    F1-Score  : {metrics['f1_score']}%")
    print(f"\n    Confusion Matrix: {metrics['confusion_matrix']}")
    print(f"\n{metrics['classification_report']}")

    # 4. Save model
    save_model(model, vectorizer)
    print("[OK] Model and vectorizer saved successfully!")
    print("     -> model.pkl")
    print("     -> vectorizer.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
