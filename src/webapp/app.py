"""
Flask Web Application for Clinical Notes Agent System.

Provides:
- File upload for clinical notes
- Real-time observability of agent activity
- Display of agent conversations and tool calls
- Results visualization
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.clinical_notes_graph import process_clinical_note
from models.graph_state import ClinicalNoteState
from utils.agent_logger import AgentLogger, get_session_activity

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")

# Configuration
UPLOAD_FOLDER = Path(__file__).parent.parent.parent / ".data" / "uploads"
RESULTS_FOLDER = Path(__file__).parent.parent.parent / ".data" / "results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'md'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Generate session ID
    session_id = str(uuid.uuid4())
    session['current_session'] = session_id

    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{filename}"
    filepath = UPLOAD_FOLDER / safe_filename

    file.save(str(filepath))

    # Read file content
    with open(filepath, 'r', encoding='utf-8') as f:
        note_text = f.read()

    # Initialize agent activity logger
    logger = AgentLogger(session_id)
    logger.log_system(
        f"Processing started for: {filename}",
        file_size=len(note_text),
        filename=filename
    )

    return jsonify({
        "session_id": session_id,
        "filename": filename,
        "status": "uploaded",
        "message": "File uploaded successfully. Starting processing..."
    })


@app.route('/process/<session_id>', methods=['POST'])
def process_note(session_id):
    """Process clinical note through agent system."""
    try:
        # Get uploaded file for this session
        files = list(UPLOAD_FOLDER.glob(f"*"))
        if not files:
            return jsonify({"error": "No uploaded file found"}), 404

        # Use most recent file (simple approach for demo)
        filepath = sorted(files, key=lambda x: x.stat().st_mtime)[-1]

        with open(filepath, 'r', encoding='utf-8') as f:
            note_text = f.read()

        logger = AgentLogger(session_id)
        logger.log_system(
            f"Starting agent workflow with {len(note_text)} characters",
            note_length=len(note_text)
        )

        # Process note through agent system
        # Note: This will block. In production, use Celery or background tasks
        collection_name = f"session_{session_id}"

        result = process_clinical_note(
            note_path=str(filepath),
            note_text=note_text,
            collection_name=collection_name,
            session_id=session_id
        )

        # Save results
        result_file = RESULTS_FOLDER / f"{session_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                "session_id": session_id,
                "toc_output": result.get("toc_output"),
                "summary_output": result.get("summary_output"),
                "recommendation_output": result.get("recommendation_output"),
                "evaluation_status": result.get("evaluation_status"),
                "iteration_count": result.get("iteration_count", 0),
                "quality_metrics": result.get("quality_metrics", {})
            }, f, indent=2)

        logger.log_system(
            f"Processing complete. Status: {result.get('evaluation_status')}",
            iterations=result.get("iteration_count", 0),
            status=result.get("evaluation_status")
        )

        return jsonify({
            "status": "completed",
            "session_id": session_id,
            "evaluation_status": result.get("evaluation_status"),
            "iteration_count": result.get("iteration_count", 0)
        })

    except Exception as e:
        logger = AgentLogger(session_id)
        logger.log_error("System", f"Processing failed: {str(e)}", error_type=type(e).__name__)
        return jsonify({"error": str(e)}), 500


@app.route('/activity/<session_id>')
def get_activity(session_id):
    """Get agent activity log for session."""
    activities = get_session_activity(session_id)
    return jsonify({
        "session_id": session_id,
        "activity_count": len(activities),
        "activities": activities
    })


@app.route('/results/<session_id>')
def get_results(session_id):
    """Get processing results for session."""
    result_file = RESULTS_FOLDER / f"{session_id}.json"

    if not result_file.exists():
        return jsonify({"error": "Results not found"}), 404

    with open(result_file, 'r') as f:
        results = json.load(f)

    return jsonify(results)


@app.route('/observability/<session_id>')
def observability(session_id):
    """Observability dashboard showing agent activity."""
    return render_template('observability.html', session_id=session_id)


if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Clinical Notes Agent - Web Application                     ║
    ╚══════════════════════════════════════════════════════════════╝

    🌐 Server: http://localhost:{port}
    📊 Observability: http://localhost:{port}/observability/<session_id>
    🔧 Debug Mode: {debug}

    📁 Upload Folder: {UPLOAD_FOLDER}
    💾 Results Folder: {RESULTS_FOLDER}

    Press Ctrl+C to stop the server
    """)

    app.run(host='0.0.0.0', port=port, debug=debug)
