"""Shared utility functions for the benchmark."""

import json
from pathlib import Path


def load_json_files(directory):
    """Load every JSON file in a directory and return parsed objects."""
    directory = Path(directory)
    results = []

    for json_path in sorted(directory.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            results.append(json.load(file))

    return results
