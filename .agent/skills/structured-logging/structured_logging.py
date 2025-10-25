"""
Structured Logging Skill.

This module provides JSON-based structured logging for audit trails and debugging.
Logs are written to daily files in logs/YYYY-MM-DD.json format.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from ..models.evaluator_schema import LogEntry, ErrorDetails, QualityMetrics


class StructuredLogger:
    """
    JSON-based structured logger for agent operations.

    Attributes:
        log_dir: Directory for log files (default: logs/)
        retention_days: Number of days to retain logs (default: 30)
    """

    def __init__(
        self,
        log_dir: str = "logs",
        retention_days: int = 30
    ):
        """
        Initialize structured logger.

        Args:
            log_dir: Directory for log files
            retention_days: Number of days to retain logs before auto-deletion
        """
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file_path(self, date: Optional[datetime] = None) -> Path:
        """
        Get log file path for a specific date.

        Args:
            date: Date for log file (default: today)

        Returns:
            Path to log file (logs/YYYY-MM-DD.json)
        """
        if date is None:
            date = datetime.utcnow()

        filename = f"{date.strftime('%Y-%m-%d')}.json"
        return self.log_dir / filename

    def log(
        self,
        log_level: str,
        agent_or_skill_name: str,
        operation_type: str,
        input_summary: Optional[str] = None,
        output_summary: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        quality_metrics: Optional[QualityMetrics] = None,
        error_details: Optional[ErrorDetails] = None
    ) -> None:
        """
        Write a structured log entry.

        Args:
            log_level: "INFO", "ERROR", "DEBUG"
            agent_or_skill_name: Source of the log
            operation_type: "invoke", "iterate", "error"
            input_summary: Abbreviated input (sanitized, no PHI)
            output_summary: Abbreviated output (sanitized, no PHI)
            execution_time_ms: Duration in milliseconds
            quality_metrics: Quality metrics if applicable
            error_details: Error information if log_level is ERROR
        """
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            log_level=log_level,
            agent_or_skill_name=agent_or_skill_name,
            operation_type=operation_type,
            input_summary=input_summary,
            output_summary=output_summary,
            execution_time_ms=execution_time_ms,
            quality_metrics=quality_metrics,
            error_details=error_details
        )

        # Get log file path for today
        log_file = self._get_log_file_path()

        # Append log entry as newline-delimited JSON
        with open(log_file, 'a') as f:
            f.write(log_entry.json() + '\n')

    def cleanup_old_logs(self) -> int:
        """
        Delete log files older than retention_days.

        Returns:
            Number of log files deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        deleted_count = 0

        for log_file in self.log_dir.glob('*.json'):
            try:
                # Parse date from filename (YYYY-MM-DD.json)
                file_date_str = log_file.stem  # Remove .json extension
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')

                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1

            except (ValueError, OSError):
                # Skip files that don't match the expected format or can't be deleted
                continue

        return deleted_count

    def read_logs(
        self,
        date: Optional[datetime] = None,
        log_level: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> list:
        """
        Read and filter log entries for a specific date.

        Args:
            date: Date to read logs from (default: today)
            log_level: Optional filter by log level
            agent_name: Optional filter by agent/skill name

        Returns:
            List of log entry dictionaries
        """
        log_file = self._get_log_file_path(date)

        if not log_file.exists():
            return []

        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Apply filters
                    if log_level and entry.get('log_level') != log_level:
                        continue
                    if agent_name and entry.get('agent_or_skill_name') != agent_name:
                        continue

                    entries.append(entry)

                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

        return entries
