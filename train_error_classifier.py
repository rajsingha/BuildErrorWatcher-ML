import os
import argparse
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a build-error classifier (TF-IDF + Logistic Regression) with validation"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/dataset.csv",
        help="Path to CSV with columns: line,label (1=error,0=noise)"
    )
    parser.add_argument(
        "--out", "-o",
        default="error_classifier.joblib",
        help="Output path for the trained joblib model"
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="Also convert and save an ONNX model alongside"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.6,
        help="Fraction of data to reserve for final testing"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.3,
        help="Fraction of training data to reserve for validation"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for splitting and algorithms"
    )
    parser.add_argument(
        "--ngram-min", type=int, default=1,
        help="Minimum n-gram size for TF-IDF"
    )
    parser.add_argument(
        "--ngram-max", type=int, default=2,
        help="Maximum n-gram size for TF-IDF"
    )
    parser.add_argument(
        "--max-features", type=int, default=5000,
        help="Max features for TF-IDF vectorizer"
    )
    parser.add_argument(
        "--solver", default="liblinear",
        choices=["liblinear", "saga", "lbfgs"],
        help="Solver for LogisticRegression"
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run GridSearchCV to tune hyperparameters"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel jobs for grid search"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        level=getattr(logging, args.log_level))

    # Load data
    if not os.path.isfile(args.data):
        logging.error(f"Data file not found: {args.data}")
        return
    df = pd.read_csv(args.data)
    X = df['line']
    y = df['label']
    logging.info(f"Loaded {len(df)} samples from {args.data}")

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    logging.info(f"Train+Val/Test split: {len(X_train_val)}/{len(X_test)} samples")

    # Further split train into train and validation
    val_frac = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac,
        random_state=args.random_state,
        stratify=y_train_val
    )
    logging.info(f"Train/Val split: {len(X_train)}/{len(X_val)} samples")

    # Build base pipeline
    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.max_features
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver=args.solver,
            max_iter=300,
            random_state=args.random_state
        ))
    ])

    pipeline = base_pipeline
    if args.grid_search:
        param_grid = {
            'tfidf__ngram_range': [
                (args.ngram_min, args.ngram_max),
                (1, 3)
            ],
            'tfidf__max_features': [args.max_features, args.max_features * 2],
            'clf__C': [0.01, 0.1, 1, 10]
        }
        pipeline = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=5,
            n_jobs=args.n_jobs,
            verbose=2
        )
        logging.info("Using GridSearchCV for hyperparameter tuning")

    # Train
    start = time.time()
    pipeline.fit(X_train, y_train)
    logging.info(f"Training completed in {time.time() - start:.2f}s")

    # Evaluate on validation set
    logging.info("Validation set evaluation:")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    logging.info(f"Validation Accuracy: {val_acc:.3f}")
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred, digits=3))

    # Final evaluation on test set
    logging.info("Test set evaluation:")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    logging.info(f"Test Accuracy: {test_acc:.3f}")
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred, digits=3))

    # Save model
    model_out = args.out
    joblib.dump(pipeline, model_out)
    logging.info(f"Saved model to {model_out}")

    # Optional ONNX export
    if args.onnx:
        trained = pipeline.best_estimator_ if hasattr(pipeline, 'best_estimator_') else pipeline
        logging.info("Converting to ONNX...")
        init_type = [("string_input", StringTensorType([None, 1]))]
        onnx_model = convert_sklearn(trained, initial_types=init_type)
        onnx_path = os.path.splitext(model_out)[0] + ".onnx"
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        logging.info(f"Saved ONNX model to {onnx_path}")

if __name__ == "__main__":
    main()
