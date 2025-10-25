"""
JSON Validation Skill.

This module provides utilities for validating JSON outputs against Pydantic schemas
to ensure data integrity and schema compliance.
"""

from typing import Dict, Any, Type, Optional, List
from pydantic import BaseModel, ValidationError
import json


class JSONValidator:
    """
    Validator for JSON outputs using Pydantic schemas.

    Provides validation, error reporting, and schema enforcement
    for all agent outputs.
    """

    @staticmethod
    def validate(
        data: Dict[str, Any],
        schema: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Validate data against a Pydantic schema.

        Args:
            data: Dictionary to validate
            schema: Pydantic model class to validate against

        Returns:
            Dict containing:
                - valid: bool - Whether validation passed
                - errors: List[str] - Validation errors (if any)
                - validated_data: Dict - Validated data with schema defaults (if valid)

        Example:
            >>> from ..models.toc_schema import TableOfContents
            >>> result = JSONValidator.validate(toc_data, TableOfContents)
            >>> if result['valid']:
            >>>     toc = result['validated_data']
        """
        try:
            # Validate and parse data using Pydantic
            validated_model = schema(**data)

            return {
                "valid": True,
                "errors": [],
                "validated_data": validated_model.dict()
            }

        except ValidationError as e:
            # Extract error messages from Pydantic validation errors
            errors = []
            for error in e.errors():
                field_path = " -> ".join(str(loc) for loc in error['loc'])
                error_msg = f"{field_path}: {error['msg']}"
                errors.append(error_msg)

            return {
                "valid": False,
                "errors": errors,
                "validated_data": None
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Unexpected validation error: {str(e)}"],
                "validated_data": None
            }

    @staticmethod
    def validate_json_string(
        json_string: str,
        schema: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Parse and validate a JSON string against a Pydantic schema.

        Args:
            json_string: JSON string to parse and validate
            schema: Pydantic model class to validate against

        Returns:
            Dict containing:
                - valid: bool - Whether validation passed
                - errors: List[str] - Parsing or validation errors
                - validated_data: Dict - Validated data (if valid)
        """
        try:
            # First, parse the JSON string
            data = json.loads(json_string)

            # Then validate against schema
            return JSONValidator.validate(data, schema)

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON parsing error: {str(e)}"],
                "validated_data": None
            }

    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """
        Extract JSON object from text that may contain additional content.

        Looks for content between first '{' and last '}' or between first '[' and last ']'.

        Args:
            text: Text that may contain JSON

        Returns:
            Extracted JSON string or None if no valid JSON found
        """
        text = text.strip()

        # Try to find JSON object
        if '{' in text and '}' in text:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx < end_idx:
                return text[start_idx:end_idx+1]

        # Try to find JSON array
        if '[' in text and ']' in text:
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx < end_idx:
                return text[start_idx:end_idx+1]

        return None

    @staticmethod
    def validate_llm_output(
        llm_output: str,
        schema: Type[BaseModel]
    ) -> Dict[str, Any]:
        """
        Extract JSON from LLM output text and validate against schema.

        Handles cases where LLM returns JSON embedded in explanatory text.

        Args:
            llm_output: Raw LLM output text that should contain JSON
            schema: Pydantic model class to validate against

        Returns:
            Dict containing:
                - valid: bool - Whether validation passed
                - errors: List[str] - Extraction or validation errors
                - validated_data: Dict - Validated data (if valid)
        """
        # Try to extract JSON from the output
        json_str = JSONValidator.extract_json_from_text(llm_output)

        if json_str is None:
            return {
                "valid": False,
                "errors": ["No JSON object or array found in LLM output"],
                "validated_data": None
            }

        # Validate the extracted JSON
        return JSONValidator.validate_json_string(json_str, schema)
