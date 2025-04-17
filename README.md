# BuildErrorWatcher-ML

**BuildErrorWatcher-ML** is a lightweight machine‑learning pipeline and CLI tool to train and export a build‑error classifier for IntelliJ IDEA plugins. It uses a TF‑IDF + Logistic Regression model to automatically distinguish real compilation or runtime errors from noisy build output lines.

---

## Features

- **Configurable pipeline**: adjust n‑gram range, vectorizer size, solver type, and more via CLI flags
- **Optional hyperparameter tuning**: run a `GridSearchCV` for automated model optimization
- **Performance logging**: detailed timing and accuracy reports on held‑out test data
- **Model persistence**: save trained pipelines as `joblib` or export to ONNX for Java/Kotlin integration

---

## Prerequisites

- Python 3.8 or newer
- `pip` for installing dependencies
- A labeled CSV dataset with two columns:
  - `line` (string): a single build‑output line
  - `label` (0=noise, 1=error)

Example dataset path: `data/build_output_lines.csv` (you can generate synthetic or capture real logs)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rajsingha/BuildErrorWatcher-ML.git
   cd ErrorSentry
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\.venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the training script to build and evaluate your classifier:

```bash
python train_error_classifier.py \
  --data data/build_output_lines.csv \
  --out error_classifier.joblib \
  --test-size 0.2 \
  --ngram-min 1 \
  --ngram-max 2 \
  --max-features 5000 \
  --solver liblinear \
  --grid-search \
  --n-jobs 4 \
  --log-level INFO
```

### Common flags

| Flag             | Description                                                     | Default                       |
| ---------------- | --------------------------------------------------------------- | ----------------------------- |
| `--data`, `-d`   | Path to your labeled CSV file                                   | `data/build_output_lines.csv` |
| `--out`, `-o`    | Destination path for the trained model (`.joblib`)              | `error_classifier.joblib`     |
| `--test-size`    | Fraction of data reserved for testing (0–1)                     | `0.2`                         |
| `--random-state` | Seed for reproducibility                                        | `42`                          |
| `--ngram-min`    | Minimum n‑gram size for TF‑IDF                                  | `1`                           |
| `--ngram-max`    | Maximum n‑gram size                                             | `2`                           |
| `--max-features` | Maximum number of TF‑IDF features                               | `5000`                        |
| `--solver`       | Solver for LogisticRegression (`liblinear`, `saga`, `lbfgs`)    | `liblinear`                   |
| `--grid-search`  | Enable hyperparameter tuning via `GridSearchCV`                 | *disabled*                    |
| `--n-jobs`       | Number of parallel jobs for grid search                         | `1`                           |
| `--onnx`         | Export trained pipeline to ONNX format alongside `.joblib` file | *disabled*                    |
| `--log-level`    | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)         | `INFO`                        |

---

## Output

- ``: serialized scikit‑learn pipeline
- `` (optional): ONNX‑formatted model for Java/Kotlin inference
- **Console report**: accuracy, precision, recall, and F1‑scores on test set

---

## Integration

1. Place `error_classifier.onnx` or `error_classifier.joblib` into your IntelliJ plugin’s `resources/` directory.
2. Load and run inference using ONNX Runtime (Java) or scikit‑learn (Python).
3. In your IntelliJ Filter implementation, call the classifier per line and highlight true errors:

```kotlin
val classifier = ErrorClassifier("/resources/error_classifier.onnx")
if (classifier.isError(line)) {
    // attach marker or gutter icon
}
```

---

## Testing

A basic `unittest` suite (`test_model.py`) verifies:

- Presence of the `.joblib` model file
- Correct predictions on representative error and noise samples
- Probability thresholds for error detection

Run tests with:

```bash
python -m unittest
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

