#!/usr/bin/env python3
"""
error_line_extractor.py

Load a trained build‑error classifier (joblib format),
read a log/text file, and write only the lines classified as errors.
"""

import argparse
import joblib
from typing import List


def main():
    p = argparse.ArgumentParser(
        description="Filter a log to only the lines classified as errors"
    )
    p.add_argument("-m", "--model", required=True,
                   help="Path to joblib model (error_classifier.joblib)")
    p.add_argument("-i", "--input", required=True,
                   help="Path to input log or text file")
    p.add_argument("-o", "--output", required=True,
                   help="Path where error lines will be written")
    args = p.parse_args()

    # load model
    clf = joblib.load(args.model)

    # read all lines
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        lines: List[str] = [ln.rstrip("\n") for ln in f]

    # predict 0/1 for each
    preds = clf.predict(lines)

    # write only lines with pred==1
    with open(args.output, "w", encoding="utf-8") as out:
        for line, flag in zip(lines, preds):
            if flag == 1:
                out.write(line + "\n")


if __name__ == "__main__":
    main()
