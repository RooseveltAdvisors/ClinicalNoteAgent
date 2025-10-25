"""
Integration test for clinical notes agent system.

Tests the end-to-end pipeline from note input to final outputs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.clinical_notes_graph import process_clinical_note


def test_basic_agent_invocation():
    """
    Test that the agent graph can be invoked with a sample note.

    This is a basic smoke test to verify:
    1. LangGraph can be instantiated
    2. State initialization works
    3. Graph can be invoked without crashing
    4. Basic flow executes (even if skills are not fully implemented)
    """
    # Sample clinical note
    test_note = """
PATIENT: John Doe
AGE: 58
CHIEF COMPLAINT: Chest pain

HISTORY OF PRESENT ILLNESS:
Patient reports sudden onset chest pain starting 1 hour ago.
Pain is substernal, 8/10 intensity, pressure-like quality.
Radiating to left arm and jaw.
Associated with diaphoresis and shortness of breath.

PAST MEDICAL HISTORY:
- Hypertension
- Hyperlipidemia
- Type 2 Diabetes

MEDICATIONS:
- Lisinopril 20mg PO daily
- Atorvastatin 40mg PO QHS
- Metformin 1000mg PO BID

ALLERGIES:
Penicillin - anaphylaxis

PHYSICAL EXAM:
Vitals: BP 160/100, HR 110, RR 22, O2 Sat 94% on RA
General: Diaphoretic, anxious, in moderate distress
Cardiovascular: Tachycardic, regular rhythm, no murmurs
Respiratory: Tachypneic, clear bilaterally

ASSESSMENT AND PLAN:
1. Acute coronary syndrome - STEMI vs NSTEMI
   - STAT EKG
   - Troponin levels q3h x 3
   - Aspirin 325mg PO stat
   - Nitroglycerin SL PRN
   - Cardiology consult STAT
   - Admit to CCU

2. Hypertension - elevated in acute setting
   - Continue home lisinopril
   - Monitor closely

3. Diabetes - hold metformin pending contrast studies
   - Insulin sliding scale
"""

    print("="*80)
    print("INTEGRATION TEST: Clinical Notes Agent System")
    print("="*80)

    print("\n[1/4] Loading test note...")
    print(f"    Note length: {len(test_note)} characters")
    print(f"    Word count: {len(test_note.split())} words")

    print("\n[2/4] Initializing agent graph...")
    try:
        from agents.clinical_notes_graph import build_clinical_notes_graph
        graph = build_clinical_notes_graph()
        print("    ✓ Graph initialized successfully")
    except Exception as e:
        print(f"    ✗ Graph initialization failed: {e}")
        return False

    print("\n[3/4] Processing clinical note...")
    try:
        result = process_clinical_note(
            note_path="/tmp/test_note.txt",
            note_text=test_note,
            collection_name="integration_test_001"
        )
        print("    ✓ Processing completed")
    except Exception as e:
        print(f"    ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[4/4] Verifying results...")
    print(f"    Evaluation status: {result.get('evaluation_status', 'N/A')}")
    print(f"    Iteration count: {result.get('iteration_count', 0)}")
    print(f"    ToC output: {'Present' if result.get('toc_output') else 'Missing'}")
    print(f"    Summary output: {'Present' if result.get('summary_output') else 'Missing'}")
    print(f"    Plan output: {'Present' if result.get('plan_output') else 'Missing'}")

    if result.get('error_occurred'):
        print(f"\n    ⚠️  Error occurred: {result.get('error_message')}")
        return False

    print("\n" + "="*80)
    print("TEST RESULT: BASIC INVOCATION SUCCESSFUL")
    print("="*80)
    print("\nNote: Skills are not fully implemented yet (TODOs in code).")
    print("This test verifies the graph structure and basic flow work correctly.")

    return True


def test_sample_notes():
    """
    Test with downloaded sample notes from .data/asclepius_notes/
    """
    notes_dir = Path(__file__).parent.parent.parent / ".data" / "asclepius_notes"

    if not notes_dir.exists():
        print(f"❌ Sample notes directory not found: {notes_dir}")
        print("Run: uv run python src/utils/download_data.py")
        return False

    note_files = list(notes_dir.glob("*.txt"))
    if not note_files:
        print(f"❌ No sample notes found in: {notes_dir}")
        return False

    print(f"\n📁 Found {len(note_files)} sample notes")
    print(f"   Directory: {notes_dir}")

    for note_file in note_files:
        print(f"\n{'='*80}")
        print(f"Testing: {note_file.name}")
        print(f"{'='*80}")

        note_text = note_file.read_text(encoding='utf-8')
        print(f"  Length: {len(note_text)} chars")

        try:
            result = process_clinical_note(
                note_path=str(note_file),
                note_text=note_text,
                collection_name=f"test_{note_file.stem}"
            )
            print(f"  ✓ Processing completed")
            print(f"  Status: {result.get('evaluation_status', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    return True


if __name__ == "__main__":
    print("\nClinical Notes Agent System - Integration Tests\n")

    # Test 1: Basic invocation
    print("\n" + "━"*80)
    print("TEST 1: Basic Agent Invocation")
    print("━"*80)
    success = test_basic_agent_invocation()

    if not success:
        print("\n⚠️  Basic invocation test failed")
        print("This is expected if Ollama/ChromaDB are not running or Phi-4 is not downloaded")
        print("\nTo fix:")
        print("  1. docker-compose up -d")
        print("  2. docker exec ollama ollama pull phi4")
        print("  3. Wait for model download to complete")
        sys.exit(1)

    # Test 2: Sample notes
    print("\n" + "━"*80)
    print("TEST 2: Sample Notes from Dataset")
    print("━"*80)
    test_sample_notes()

    print("\n✅ All tests completed!")
