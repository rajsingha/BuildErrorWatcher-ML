#!/usr/bin/env python3
import os
import joblib
import re
from typing import List

class ErrorLogFilter:
    """
    Load a line‑level error classifier (joblib) and filter raw log files,
    printing only the lines predicted as errors (or matching known error patterns).
    """
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.pipeline = joblib.load(model_path)

        # Heuristic error patterns
        self.error_patterns = [
            # Compilation errors
            r'error[:\s]',
            r'exception in',
            r'failed with exit code',
            r'compilation failed',
            r'build failed',
            r'unresolved reference',
            # Common Java/JVM errors
            r'nullpointerexception',
            r'classnotfoundexception',
            r'outofmemoryerror',
            r'stackoverflowerror',
            # JavaScript/Node errors
            r'cannot find module',
            r'unexpected token',
            r'is not defined',
            r'is not a function',
            # Python errors
            r'importerror',
            r'indentationerror',
            r'syntaxerror',
            r'nameerror',
            # Build tool errors
            r'could not resolve',
            r'dependency not found',
            r'failed to resolve',
            # Docker/container errors
            r'image not found',
            r'container exited',
        ]
        self._regexes = [re.compile(pat, re.IGNORECASE) for pat in self.error_patterns]

    def _load_lines(self, log_path: str) -> List[str]:
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.rstrip("\n") for line in f]

    def _is_error(self, line: str) -> bool:
        # first check heuristics
        for rx in self._regexes:
            if rx.search(line):
                return True
        # fallback to trained model
        return bool(self.pipeline.predict([line])[0])

    def extract_error_lines(self, lines: List[str]) -> List[str]:
        """
        Return only those lines which are either matched by heuristics
        or labeled as errors by the model.
        """
        return [line for line in lines if self._is_error(line)]

    def print_errors_from_file(self, log_path: str):
        """
        Load a log file, filter for error lines, and print them.
        """
        lines = self._load_lines(log_path)
        for err in self.extract_error_lines(lines):
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
        default="build-error.txt",
        help="Path to the raw build log file (e.g. build_output.txt)"
    )
    args = parser.parse_args()

    filt = ErrorLogFilter(args.model)
    filt.print_errors_from_file(args.log)
