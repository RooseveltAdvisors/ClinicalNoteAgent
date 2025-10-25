"""
File Cleanup Skill.

This module handles deletion of temporary session files and expired logs
to maintain privacy and prevent disk space exhaustion.
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class FileCleanup:
    """
    Utility for cleaning up temporary files and session data.

    Attributes:
        data_dir: Directory for temporary session files (default: .data/sessions/)
        log_dir: Directory for logs (default: logs/)
    """

    def __init__(
        self,
        data_dir: str = ".data/sessions",
        log_dir: str = "logs"
    ):
        """
        Initialize file cleanup utility.

        Args:
            data_dir: Directory containing temporary session files
            log_dir: Directory containing log files
        """
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)

    def delete_session_files(self, session_id: Optional[str] = None) -> int:
        """
        Delete session-specific files (uploaded notes, generated outputs).

        Args:
            session_id: Optional session ID to delete (if None, deletes all sessions)

        Returns:
            Number of files deleted
        """
        if not self.data_dir.exists():
            return 0

        deleted_count = 0

        if session_id:
            # Delete specific session directory
            session_dir = self.data_dir / session_id
            if session_dir.exists() and session_dir.is_dir():
                for file_path in session_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except OSError:
                            pass
                # Remove empty session directory
                try:
                    session_dir.rmdir()
                except OSError:
                    pass
        else:
            # Delete all session directories
            for session_dir in self.data_dir.iterdir():
                if session_dir.is_dir():
                    for file_path in session_dir.rglob('*'):
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                                deleted_count += 1
                            except OSError:
                                pass
                    # Remove empty session directory
                    try:
                        session_dir.rmdir()
                    except OSError:
                        pass

        return deleted_count

    def delete_file(self, file_path: str) -> bool:
        """
        Delete a specific file safely.

        Args:
            file_path: Path to file to delete

        Returns:
            bool: True if file was deleted, False otherwise
        """
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                return True
            return False
        except OSError:
            return False

    def delete_expired_logs(self, retention_days: int = 30) -> int:
        """
        Delete log files older than retention_days.

        Args:
            retention_days: Number of days to retain logs

        Returns:
            Number of log files deleted
        """
        # Import here to avoid circular dependency
        from .structured_logging import StructuredLogger

        logger = StructuredLogger(log_dir=str(self.log_dir), retention_days=retention_days)
        return logger.cleanup_old_logs()

    def create_session_dir(self, session_id: str) -> Path:
        """
        Create a session directory for temporary files.

        Args:
            session_id: Unique session identifier

        Returns:
            Path to created session directory
        """
        session_dir = self.data_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def list_session_files(self, session_id: str) -> List[str]:
        """
        List all files in a session directory.

        Args:
            session_id: Session identifier

        Returns:
            List of file paths in the session
        """
        session_dir = self.data_dir / session_id

        if not session_dir.exists():
            return []

        return [str(f) for f in session_dir.rglob('*') if f.is_file()]
