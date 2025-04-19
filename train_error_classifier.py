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
        "--test-size", type=float, default=0.2,
        help="Fraction of data to reserve for final testing"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2,
        help="Fraction of training data to reserve for validation"
    )
    parser.add_argument(
        "--random-state", type=int, default=59,
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
        "--max-features", type=int, default=3000,
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
    parser.add_argument(
        "--regularization", type=float, default=1.0,
        help="C parameter for logistic regression (lower values = stronger regularization)"
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

    # First split to get a test set aside
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Then split the temp set into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_temp
    )

    logging.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")

    # Apply basic text cleaning (remove duplicates in train set)
    X_train_unique = X_train.drop_duplicates()
    y_train_unique = y_train.loc[X_train_unique.index]
    duplicate_reduction = len(X_train) - len(X_train_unique)
    if duplicate_reduction > 0:
        logging.info(f"Removed {duplicate_reduction} duplicate entries from training data")
        X_train = X_train_unique
        y_train = y_train_unique

    # Build base pipeline with increased regularization to combat overfitting
    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.max_features,
            min_df=3,  # Ignore terms that appear in less than 3 documents
            max_df=0.9  # Ignore terms that appear in more than 90% of documents
        )),
        ("clf", LogisticRegression(
            C=args.regularization,  # Lower C means more regularization
            class_weight="balanced",
            solver=args.solver,
            max_iter=500,
            random_state=args.random_state
        ))
    ])

    pipeline = base_pipeline
    if args.grid_search:
        param_grid = {
            'tfidf__max_features': [2000, 3000, 5000],
            'tfidf__min_df': [2, 3, 5],
            'tfidf__max_df': [0.8, 0.9, 0.95],
            'clf__C': [0.01, 0.1, 1.0]  # Try different regularization strengths
        }
        pipeline = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=5,
            n_jobs=args.n_jobs,
            verbose=2,
            scoring='f1_weighted'  # Use F1 score for optimization
        )
        logging.info("Using GridSearchCV for hyperparameter tuning")

    # Train
    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start
    logging.info(f"Training completed in {training_time:.2f}s")

    if args.grid_search and hasattr(pipeline, 'best_params_'):
        logging.info(f"Best parameters: {pipeline.best_params_}")

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
