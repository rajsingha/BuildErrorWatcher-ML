import os
import argparse
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import hashlib
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a build-error classifier (TF-IDF + Logistic Regression) with robust validation"
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
        "--plots-dir",
        default="model_evaluation",
        help="Directory to save evaluation plots"
    )
    parser.add_argument(
        "--onnx", action="store_true",
        help="Also convert and save an ONNX model alongside"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.3,
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
        "--max-features", type=int, default=3000,
        help="Max features for TF-IDF vectorizer"
    )
    parser.add_argument(
        "--min-df", type=int, default=5,
        help="Minimum document frequency for TF-IDF features"
    )
    parser.add_argument(
        "--max-df", type=float, default=0.8,
        help="Maximum document frequency for TF-IDF features (0.0-1.0)"
    )
    parser.add_argument(
        "--solver", default="liblinear",
        choices=["liblinear", "saga", "lbfgs"],
        help="Solver for LogisticRegression"
    )
    parser.add_argument(
        "--c-value", type=float, default=0.1,
        help="Regularization strength (C parameter) for LogisticRegression (default lowered to 0.1)"
    )
    parser.add_argument(
        "--l1-ratio", type=float, default=0.0,
        help="L1 ratio for Elastic-Net regularization (0=L2, 1=L1, in-between=mix)"
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run GridSearchCV to tune hyperparameters"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs for grid search (-1 for all CPUs)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--clean-text", action="store_true", default=True,
        help="Enable text cleaning and normalization"
    )
    parser.add_argument(
        "--deduplicate-first", action="store_true", default=True,
        help="Deduplicate the dataset before splitting"
    )
    return parser.parse_args()


def setup_logging(log_level, log_file="error_classifier_training.log"):
    """Set up logging to both console and file with proper encoding"""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with ASCII only
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def clean_text(text):
    """Apply text cleaning to reduce overfitting on irrelevant patterns"""
    if pd.isna(text):
        return ""

    # Convert to string if not already
    text = str(text)

    # Strip whitespace
    text = text.strip()

    # Normalize whitespace (replace multiple spaces with single space)
    text = re.sub(r'\s+', ' ', text)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Lowercase text
    text = text.lower()

    # Remove special characters except basic punctuation
    # text = re.sub(r'[^\w\s\.\,\:\;\!\?]', '', text)

    return text


def compute_text_hash(text):
    """Create a hash of text content to identify duplicates more efficiently"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def preprocess_data(df, clean_texts=True):
    """Clean and preprocess the dataset"""
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Check for and handle missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logging.warning(f"Found {missing_values.sum()} missing values: {missing_values}")
        df = df.dropna()
        logging.info(f"Dropped rows with missing values, new shape: {df.shape}")

    # Clean text content if requested
    if clean_texts and 'line' in df.columns:
        logging.info("Applying text cleaning to reduce noise and improve generalization")
        df['line'] = df['line'].apply(clean_text)

        # Remove completely empty lines
        empty_mask = df['line'].str.len() == 0
        if empty_mask.sum() > 0:
            logging.info(f"Removing {empty_mask.sum()} empty lines")
            df = df[~empty_mask]

    # Add a hash column for deduplication efficiency
    if 'line' in df.columns:
        df['text_hash'] = df['line'].apply(compute_text_hash)

    return df


def deduplicate_dataset(df):
    """Remove exact duplicates from the entire dataset before splitting"""
    original_size = len(df)

    # First check if there are duplicates
    if df['text_hash'].duplicated().sum() == 0:
        logging.info("No duplicates found in dataset")
        return df

    # Group by text hash and keep one example from each group
    # For duplicate texts, maintain label consistency by majority vote
    grouped = df.groupby('text_hash')
    deduped_rows = []

    for _, group in grouped:
        if len(group) == 1:
            # No duplicates for this text
            deduped_rows.append(group.iloc[0])
        else:
            # Duplicates found, use majority vote for label
            majority_label = group['label'].mode()[0]
            rep_row = group.iloc[0].copy()
            rep_row['label'] = majority_label
            deduped_rows.append(rep_row)

    deduped_df = pd.DataFrame(deduped_rows)

    # Report on deduplication
    removed = original_size - len(deduped_df)
    logging.info(f"Removed {removed} duplicate texts ({removed / original_size:.1%} of dataset)")

    # Look for label inconsistencies
    inconsistencies = original_size - removed - len(deduped_df)
    if inconsistencies > 0:
        logging.warning(f"Found {inconsistencies} texts with inconsistent labels (resolved by majority vote)")

    return deduped_df


def check_data_leakage(X_train, X_val, X_test):
    """Check for potential data leakage between sets"""
    train_set = set(X_train)
    val_set = set(X_val)
    test_set = set(X_test)

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    if train_val_overlap > 0:
        logging.warning(f"LEAKAGE DETECTED: {train_val_overlap} duplicate samples between train and validation sets")
    if train_test_overlap > 0:
        logging.warning(f"LEAKAGE DETECTED: {train_test_overlap} duplicate samples between train and test sets")
    if val_test_overlap > 0:
        logging.warning(f"LEAKAGE DETECTED: {val_test_overlap} duplicate samples between validation and test sets")

    return {
        'train_val_overlap': train_val_overlap,
        'train_test_overlap': train_test_overlap,
        'val_test_overlap': val_test_overlap
    }


def split_with_no_leakage(df, test_size, val_size, random_state):
    """Create train/val/test splits ensuring no data leakage"""
    # We'll use text_hash for deduplication
    if 'text_hash' not in df.columns:
        df['text_hash'] = df['line'].apply(compute_text_hash)

    # Get unique text hashes to avoid leakage
    unique_hashes = df['text_hash'].unique()
    n_samples = len(unique_hashes)

    # Calculate split sizes
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size * (1 - test_size))

    # Generate indices for splits
    indices = np.random.RandomState(random_state).permutation(n_samples)
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]

    # Get the text hashes for each split
    test_hashes = unique_hashes[test_indices]
    val_hashes = unique_hashes[val_indices]
    train_hashes = unique_hashes[train_indices]

    # Create masks based on hash membership
    test_mask = df['text_hash'].isin(test_hashes)
    val_mask = df['text_hash'].isin(val_hashes)
    train_mask = df['text_hash'].isin(train_hashes)

    # Create the splits
    df_test = df[test_mask]
    df_val = df[val_mask]
    df_train = df[train_mask]

    # Ensure stratification by rebalancing if needed
    for split_df in [df_train, df_val, df_test]:
        # Get current class balance
        class_counts = split_df['label'].value_counts()
        logging.debug(f"Class distribution in split: {dict(class_counts)}")

    return df_train, df_val, df_test


def create_visualizations(pipeline, X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """Generate and save evaluation visualizations"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get predicted probabilities
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Prediction histograms by class
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(y_train_proba[y_train == 0], color='blue', alpha=0.5, bins=20, label='Class 0')
    sns.histplot(y_train_proba[y_train == 1], color='red', alpha=0.5, bins=20, label='Class 1')
    plt.title('Training Set Predictions')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(1, 3, 2)
    sns.histplot(y_val_proba[y_val == 0], color='blue', alpha=0.5, bins=20, label='Class 0')
    sns.histplot(y_val_proba[y_val == 1], color='red', alpha=0.5, bins=20, label='Class 1')
    plt.title('Validation Set Predictions')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Count')
    plt.legend()

    plt.subplot(1, 3, 3)
    sns.histplot(y_test_proba[y_test == 0], color='blue', alpha=0.5, bins=20, label='Class 0')
    sns.histplot(y_test_proba[y_test == 1], color='red', alpha=0.5, bins=20, label='Class 1')
    plt.title('Test Set Predictions')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Count')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distributions.png'))
    plt.close()

    # Create confusion matrix visualization
    for name, y_true, y_pred in [
        ('train', y_train, pipeline.predict(X_train)),
        ('validation', y_val, pipeline.predict(X_val)),
        ('test', y_test, pipeline.predict(X_test))
    ]:
        plt.figure(figsize=(8, 6))
        cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name.capitalize()} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'))
        plt.close()

    # Extract and analyze feature importances
    if hasattr(pipeline, 'best_estimator_'):
        final_model = pipeline.best_estimator_
    else:
        final_model = pipeline

    if isinstance(final_model[-1], LogisticRegression):
        try:
            # Get TF-IDF feature names
            tfidf_vectorizer = final_model[0]
            feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

            # Get model coefficients
            coef = final_model[-1].coef_[0]

            # Create DataFrame for feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)

            # Plot top positive and negative features
            plt.figure(figsize=(12, 8))
            top_n = min(20, len(feature_importance))
            top_features = feature_importance.head(top_n)

            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title(f'Top {top_n} Important Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
            plt.close()

            # Save all feature importances to CSV
            feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        except Exception as e:
            logging.warning(f"Could not generate feature importance visualization: {e}")


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Load data
    if not os.path.isfile(args.data):
        logging.error(f"Data file not found: {args.data}")
        return

    df = pd.read_csv(args.data)
    logging.info(f"Loaded raw dataset with shape: {df.shape}")

    # Check for required columns
    if 'line' not in df.columns or 'label' not in df.columns:
        logging.error(f"Dataset must have 'line' and 'label' columns. Found: {df.columns.tolist()}")
        return

    # Data preprocessing
    df = preprocess_data(df, clean_texts=args.clean_text)

    # Check class balance before deduplication
    class_counts_before = Counter(df['label'])
    logging.info(f"Class distribution before deduplication: {class_counts_before}")

    # Deduplicate the dataset BEFORE splitting to prevent leakage
    if args.deduplicate_first:
        df = deduplicate_dataset(df)

    # Check class balance after deduplication
    class_counts_after = Counter(df['label'])
    logging.info(f"Class distribution after deduplication: {class_counts_after}")

    # Create train/val/test splits ensuring no leakage
    df_train, df_val, df_test = split_with_no_leakage(df, args.test_size, args.val_size, args.random_state)

    # Extract features and labels
    X_train, y_train = df_train['line'], df_train['label']
    X_val, y_val = df_val['line'], df_val['label']
    X_test, y_test = df_test['line'], df_test['label']

    logging.info(f"Final split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Verify no leakage in the splits
    leakage_info = check_data_leakage(X_train, X_val, X_test)
    if any(leakage_info.values()):
        logging.error("Split strategy failed to eliminate data leakage!")
        return

    # Build pipeline with strong regularization to combat overfitting
    logging.info("Building model pipeline with regularization...")

    # Use saga solver for mixed L1/L2 regularization
    if args.l1_ratio > 0:
        penalty = 'elasticnet'
        solver = 'saga'
    else:
        penalty = 'l2'
        solver = args.solver

    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.max_features,
            min_df=args.min_df,
            max_df=args.max_df,
            sublinear_tf=True,  # Apply log scaling to term frequencies
            stop_words='english'  # Remove English stop words
        )),
        ("clf", LogisticRegression(
            C=args.c_value,
            penalty=penalty,
            solver=solver,
            l1_ratio=args.l1_ratio if penalty == 'elasticnet' else None,
            class_weight="balanced",
            max_iter=1000,
            random_state=args.random_state,
            n_jobs=1  # Single job for predictable behavior
        ))
    ])

    pipeline = base_pipeline
    if args.grid_search:
        param_grid = {
            'tfidf__max_features': [2000, 3000, 5000],
            'tfidf__min_df': [3, 5, 10],
            'tfidf__max_df': [0.7, 0.8, 0.9],
            'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0],
        }

        # Add elasticnet parameters if using saga
        if solver == 'saga':
            param_grid['clf__l1_ratio'] = [0.1, 0.5, 0.9]

        # Create stratified CV with shuffling
        cv = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.random_state
        )

        pipeline = GridSearchCV(
            base_pipeline,
            param_grid=param_grid,
            cv=cv,
            n_jobs=args.n_jobs,
            verbose=2,
            scoring='f1_weighted',
            return_train_score=True
        )
        logging.info(f"Using GridSearchCV with {args.cv_folds}-fold CV for hyperparameter tuning")

    # Train
    start = time.time()
    try:
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start
        logging.info(f"Training completed in {training_time:.2f} seconds")

        if args.grid_search and hasattr(pipeline, 'best_params_'):
            logging.info(f"Best parameters: {pipeline.best_params_}")
            logging.info(f"Cross-validation results summary:")
            logging.info(f"Best CV score: {pipeline.best_score_:.4f}")

            # Calculate CV score ranges
            cv_results = pipeline.cv_results_
            best_idx = pipeline.best_index_
            cv_scores = []
            for i in range(args.cv_folds):
                score_key = f'split{i}_test_score'
                if score_key in cv_results:
                    cv_scores.append(cv_results[score_key][best_idx])

            if cv_scores:
                logging.info(f"CV score range: {min(cv_scores):.4f} - {max(cv_scores):.4f}")
                logging.info(f"CV score std: {np.std(cv_scores):.4f}")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return

    # Evaluate on training set (to check for overfitting)
    logging.info("Training set evaluation:")
    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    logging.info(f"Training Accuracy: {train_acc:.4f}")
    logging.info(f"Training F1 Score: {train_f1:.4f}")

    # Evaluate on validation set
    logging.info("Validation set evaluation:")
    y_val_pred = pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    logging.info(f"Validation Accuracy: {val_acc:.4f}")
    logging.info(f"Validation F1 Score: {val_f1:.4f}")
    print("Validation Classification Report:\n", classification_report(y_val, y_val_pred, digits=4))

    # Calculate validation AUC if binary classification
    if len(np.unique(y_train)) == 2:
        val_auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        logging.info(f"Validation AUC: {val_auc:.4f}")

    # Final evaluation on test set
    logging.info("Test set evaluation:")
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred, digits=4))

    # Calculate test AUC if binary classification
    if len(np.unique(y_train)) == 2:
        test_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        logging.info(f"Test AUC: {test_auc:.4f}")

    # Check for overfitting
    acc_diff = train_acc - val_acc
    if acc_diff > 0.05:
        logging.warning(f"Potential overfitting detected: Train-Val accuracy gap is {acc_diff:.4f}")
        if acc_diff > 0.2:
            logging.warning("Severe overfitting detected! Consider stronger regularization.")

    if train_acc > 0.98 and val_acc > 0.98:
        logging.warning("Near-perfect scores may indicate remaining issues with dataset")

    # Create and save visualizations
    try:
        create_visualizations(pipeline, X_train, y_train, X_val, y_val, X_test, y_test, args.plots_dir)
        logging.info(f"Evaluation visualizations saved to {args.plots_dir}")
    except Exception as e:
        logging.warning(f"Could not generate visualizations: {e}")

    # Save model
    model_out = args.out
    try:
        joblib.dump(pipeline, model_out)
        logging.info(f"Saved model to {model_out}")

        # Save model metadata
        metadata = {
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_size': {
                'original': int(df.shape[0]),
                'after_deduplication': int(len(df)),
                'training': int(len(X_train)),
                'validation': int(len(X_val)),
                'test': int(len(X_test))
            },
            'class_balance': {
                'original': {str(k): int(v) for k, v in class_counts_before.items()},
                'after_deduplication': {str(k): int(v) for k, v in class_counts_after.items()}
            },
            'metrics': {
                'training_accuracy': float(train_acc),
                'validation_accuracy': float(val_acc),
                'test_accuracy': float(test_acc),
                'training_f1': float(train_f1),
                'validation_f1': float(val_f1),
                'test_f1': float(test_f1)
            },
            'parameters': vars(args)
        }

        if len(np.unique(y_train)) == 2:
            metadata['metrics']['validation_auc'] = float(val_auc)
            metadata['metrics']['test_auc'] = float(test_auc)

        metadata_file = os.path.splitext(model_out)[0] + "_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        logging.info(f"Saved model metadata to {metadata_file}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

        # ONNX export
        try:
            trained = pipeline.best_estimator_ if hasattr(pipeline, 'best_estimator_') else pipeline
            logging.info("Converting to ONNX...")
            init_type = [("string_input", StringTensorType([None, 1]))]
            onnx_model = convert_sklearn(trained, initial_types=init_type)
            onnx_path = os.path.splitext(model_out)[0] + ".onnx"
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            logging.info(f"Saved ONNX model to {onnx_path}")
        except Exception as e:
            logging.error(f"Error during ONNX conversion: {e}")


if __name__ == "__main__":
    main()
