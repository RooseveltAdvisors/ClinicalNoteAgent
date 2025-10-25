#!/usr/bin/env python3
"""
Prepare test data from Asclepius synthetic clinical notes dataset.

This script extracts sample clinical notes from data/synthetic.csv and saves them
as individual .txt files for testing the agent system.
"""

import csv
import sys
from pathlib import Path


def extract_sample_notes(
    csv_path: Path,
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """
    Extract sample clinical notes from CSV and save as text files.

    Args:
        csv_path: Path to synthetic.csv
        output_dir: Directory to save extracted notes
        num_samples: Number of sample notes to extract
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading clinical notes from: {csv_path}")
    print(f"Extracting {num_samples} samples to: {output_dir}")

    extracted_count = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                if extracted_count >= num_samples:
                    break

                patient_id = row['patient_id']
                note_text = row['note']

                # Skip empty notes
                if not note_text.strip():
                    continue

                # Save note as text file
                output_file = output_dir / f"note_{patient_id}.txt"
                output_file.write_text(note_text, encoding='utf-8')

                print(f"✓ Extracted note {extracted_count + 1}: {output_file.name} ({len(note_text)} chars)")
                extracted_count += 1

        print(f"\n✅ Successfully extracted {extracted_count} clinical notes")

    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_path}")
        print("Please ensure data/synthetic.csv exists")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error extracting notes: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "synthetic.csv"
    output_dir = project_root / "tests" / "fixtures" / "sample_notes"

    # Extract samples
    extract_sample_notes(
        csv_path=csv_path,
        output_dir=output_dir,
        num_samples=5,
    )
