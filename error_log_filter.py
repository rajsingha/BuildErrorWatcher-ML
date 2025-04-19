import joblib
from typing import List
import os

class ErrorLogFilter:
    """
    Load a line‑level error classifier (joblib) and filter raw log files,
    printing only the lines predicted as errors.
    """
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.pipeline = joblib.load(model_path)

    def _load_lines(self, log_path: str) -> List[str]:
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        with open(log_path, "r", encoding="utf‑8", errors="ignore") as f:
            return [line.rstrip("\n") for line in f]

    def extract_error_lines(self, lines: List[str]) -> List[str]:
        """
        Run the pipeline on each line and return only those labeled as errors.
        """
        preds = self.pipeline.predict(lines)
        return [line for line, p in zip(lines, preds) if p == 1]

    def print_errors_from_file(self, log_path: str):
        """
        Load a log file, filter for error lines, and print them.
        """
        lines = self._load_lines(log_path)
        error_lines = self.extract_error_lines(lines)
        for err in error_lines:
            print(err)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print only the error lines from a build log"
    )
    parser.add_argument(
        "-m", "--model",
        default="error_classifier.joblib",
        help="Path to your trained joblib model (e.g. error_classifier.joblib)"
    )
    parser.add_argument(
        "-l", "--log",
        default="input_errors.txt",
        help="Path to the raw build log file (e.g. build_output.txt)"
    )
    args = parser.parse_args()

    filt = ErrorLogFilter(args.model)
    filt.print_errors_from_file(args.log)
