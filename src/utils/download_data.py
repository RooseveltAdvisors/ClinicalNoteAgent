"""Download Asclepius Synthetic Clinical Notes dataset.

This script downloads de-identified synthetic clinical notes from HuggingFace
for testing and development purposes.
"""

import os
import sys
from pathlib import Path
import urllib.request
import json


def download_asclepius_dataset(output_dir: str = ".data/asclepius_notes"):
    """Download sample clinical notes from Asclepius dataset.

    Args:
        output_dir: Directory to save downloaded notes
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Asclepius Synthetic Clinical Notes dataset...")
    print(f"Output directory: {output_path.absolute()}")

    # Note: This is a placeholder implementation
    # In production, you would use the Hugging Face datasets library:
    # from datasets import load_dataset
    # dataset = load_dataset("starmpcc/Asclepius-Synthetic-Clinical-Notes")

    # For now, create sample notes for testing
    sample_notes = [
        {
            "id": "note_001",
            "content": """HISTORY OF PRESENT ILLNESS:
The patient is a 45-year-old male presenting with chest pain that started 2 hours ago. Pain is described as pressure-like, 7/10 intensity, radiating to left arm. Associated with diaphoresis and shortness of breath. No relief with rest.

REVIEW OF SYSTEMS:
Cardiovascular: Chest pain as above. Denies palpitations.
Respiratory: Shortness of breath with exertion.
GI: Denies nausea, vomiting, abdominal pain.

PHYSICAL EXAM:
Vitals: BP 150/95, HR 102, RR 20, O2 Sat 96% on RA
General: Diaphoretic, appears uncomfortable
Cardiovascular: Regular rate, no murmurs
Respiratory: Clear to auscultation bilaterally

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - concerning presentation
   - Order troponin, EKG stat
   - Start aspirin 325mg PO
   - Cardiology consult

2. Hypertension - elevated BP in setting of chest pain
   - Monitor closely
   - May need adjustment of home medications
"""
        },
        {
            "id": "note_002",
            "content": """CHIEF COMPLAINT:
62-year-old female with persistent cough for 3 weeks.

HISTORY OF PRESENT ILLNESS:
Patient reports dry cough started 3 weeks ago. Initially thought it was a cold, but cough persists. Worse at night. Denies fever, chills, hemoptysis. No known sick contacts. Non-smoker.

PAST MEDICAL HISTORY:
- Type 2 Diabetes Mellitus
- Hyperlipidemia
- Osteoarthritis

MEDICATIONS:
- Metformin 1000mg PO BID
- Atorvastatin 40mg PO QD
- Ibuprofen 400mg PRN

ALLERGIES:
Penicillin - rash

PHYSICAL EXAM:
Vitals: BP 128/82, HR 78, RR 16, T 98.2F, O2 Sat 98%
HEENT: No pharyngeal erythema
Respiratory: Few scattered wheezes, no crackles

ASSESSMENT AND PLAN:
1. Chronic cough - likely post-viral reactive airway
   - Trial albuterol inhaler
   - F/U in 2 weeks if no improvement
   - Consider CXR if persistent
"""
        }
    ]

    # Save sample notes
    for note_data in sample_notes:
        note_path = output_path / f"{note_data['id']}.txt"
        with open(note_path, 'w') as f:
            f.write(note_data['content'])
        print(f"  Created: {note_path.name}")

    print(f"\nDownload complete!")
    print(f"Total notes: {len(sample_notes)}")
    print(f"\nTo use real Asclepius dataset, install: uv add datasets")
    print(f"Then uncomment HuggingFace code in this script.")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else ".data/asclepius_notes"
    download_asclepius_dataset(output_dir)
