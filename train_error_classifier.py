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
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a build-error classifier (TF-IDF + Logistic Regression)"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/build_output_lines.csv",
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
        help="Fraction of data to reserve for testing (0.0-1.0)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for train-test split and any randomized algorithms"
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

    # 1. Load data
    if not os.path.isfile(args.data):
        logging.error(f"Data file not found: {args.data}")
        return
    df = pd.read_csv(args.data)
    X = df['line']
    y = df['label']
    logging.info(f"Loaded {len(df)} samples from {args.data}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )
    logging.info(f"Train/test split: {len(X_train)}/{len(X_test)} samples")

    # 3. Build pipeline
    pipeline = Pipeline([
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

    # 4. Optional grid search
    if args.grid_search:
        param_grid = {
            'tfidf__ngram_range': [
                (args.ngram_min, args.ngram_max),
                (1,3)
            ],
            'tfidf__max_features': [args.max_features, args.max_features*2],
            'clf__C': [0.1, 1, 10]
        }
        pipeline = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            n_jobs=args.n_jobs,
            verbose=2
        )
        logging.info("Running GridSearchCV...")

    # 5. Train
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    logging.info(f"Training completed in {elapsed:.2f} seconds")

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy on test set: {acc:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # 7. Save model
    joblib.dump(pipeline, args.out)
    logging.info(f"Saved trained model to {args.out}")

    # 8. Optional ONNX export
    if args.onnx:
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import StringTensorType
            logging.info("Converting pipeline to ONNX format...")
            initial_type = [("string_input", StringTensorType([None, 1]))]
            onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
            onnx_path = os.path.splitext(args.out)[0] + ".onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logging.info(f"Saved ONNX model to {onnx_path}")
        except ImportError:
            logging.warning("skl2onnx not installed; skipping ONNX export")


if __name__ == "__main__":
    main()
