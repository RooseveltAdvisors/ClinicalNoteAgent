"""
Agent Activity Logger for Observability.

Provides centralized logging of agent activities including:
- Agent messages
- Tool calls and results
- State updates
- System events

Used by both clinical_notes_graph.py and webapp for real-time observability.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Global activity storage (in-memory for demo, could be Redis/DB in production)
_activity_storage: Dict[str, List[Dict]] = {}


class AgentLogger:
    """Logger for agent activities with session isolation."""

    def __init__(self, session_id: str):
        """Initialize logger for a specific session."""
        self.session_id = session_id
        if session_id not in _activity_storage:
            _activity_storage[session_id] = []

    def log(
        self,
        activity_type: str,
        agent_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an agent activity.

        Args:
            activity_type: Type of activity (message, tool_call, tool_result, state_update, system, error)
            agent_name: Name of the agent (toc_agent, summary_agent, etc.)
            content: Activity content/description
            metadata: Additional metadata dict
        """
        activity = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "agent": agent_name,
            "content": content,
            "metadata": metadata or {}
        }

        _activity_storage[self.session_id].append(activity)

        # Keep only last 1000 entries to prevent memory issues
        if len(_activity_storage[self.session_id]) > 1000:
            _activity_storage[self.session_id] = _activity_storage[self.session_id][-1000:]

    def log_message(self, agent_name: str, message: str, **metadata):
        """Log an agent message."""
        self.log("message", agent_name, message, metadata)

    def log_tool_call(self, agent_name: str, tool_name: str, tool_input: Any, **metadata):
        """Log a tool call."""
        content = f"Calling tool: {tool_name}"
        metadata.update({"tool_name": tool_name, "tool_input": str(tool_input)[:200]})
        self.log("tool_call", agent_name, content, metadata)

    def log_tool_result(self, agent_name: str, tool_name: str, result: Any, **metadata):
        """Log a tool result."""
        content = f"Tool {tool_name} completed"
        result_str = str(result)[:200] if result else "None"
        metadata.update({"tool_name": tool_name, "result_preview": result_str})
        self.log("tool_result", agent_name, content, metadata)

    def log_state_update(self, agent_name: str, update_description: str, **metadata):
        """Log a state update."""
        self.log("state_update", agent_name, update_description, metadata)

    def log_system(self, message: str, **metadata):
        """Log a system event."""
        self.log("system", "System", message, metadata)

    def log_error(self, agent_name: str, error_message: str, **metadata):
        """Log an error."""
        self.log("error", agent_name, error_message, metadata)

    def get_activities(self) -> List[Dict]:
        """Get all activities for this session."""
        return _activity_storage.get(self.session_id, [])

    def clear(self):
        """Clear activities for this session."""
        if self.session_id in _activity_storage:
            _activity_storage[self.session_id] = []


def get_session_activity(session_id: str) -> List[Dict]:
    """Get activities for a specific session (for webapp API)."""
    return _activity_storage.get(session_id, [])


def clear_session_activity(session_id: str):
    """Clear activities for a specific session."""
    if session_id in _activity_storage:
        del _activity_storage[session_id]
