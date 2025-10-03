"""
VSTEP JSON Schema Definition using Pydantic
Defines the exact structure for consistent VSTEP assessment outputs
"""

from pydantic import BaseModel, Field
from typing import Optional
import json


class VSTEPAssessment(BaseModel):
    """
    VSTEP Speaking Assessment JSON Schema
    Ensures consistent output format for all evaluations
    """
    grammar: str = Field(
        ..., 
        description="Detailed comment about grammar range and control with specific examples from transcript (50-100 words)",
        min_length=100,
        max_length=2000
    )
    
    vocabulary: str = Field(
        ..., 
        description="Detailed comment about vocabulary range and appropriateness with specific examples (50-100 words)",
        min_length=50,
        max_length=2000
    )
    
    discourse: str = Field(
        ..., 
        description="Detailed comment about discourse management with specific examples of organization (50-100 words)",
        min_length=50,
        max_length=2000
    )
    
    total: str = Field(
        ..., 
        description="Comprehensive overall assessment with band estimation and specific suggestions (50-100 words)",
        min_length=50,
        max_length=2000
    )
    
    class Config:
        # Example for validation
        schema_extra = {
            "example": {
                "grammar": "The candidate demonstrates good control of basic grammatical structures with occasional errors that do not impede understanding. Complex sentences show some accuracy issues.",
                "vocabulary": "Uses a range of vocabulary appropriate for familiar topics. Some repetition observed but generally adequate lexical variety for the task.",
                "discourse": "Ideas are developed with reasonable coherence and basic linking devices. Some organization evident though transitions could be smoother.",
                "total": "Overall solid performance showing competent English speaking ability with room for improvement in complex structures and fluency."
            }
        }


def get_vstep_json_schema():
    """
    Returns the JSON schema string for VSTEP assessment
    Used by Outlines for constrained generation
    """
    return json.dumps(VSTEPAssessment.model_json_schema())


def validate_vstep_output(output_dict):
    """
    Validates if the output matches VSTEP schema
    """
    try:
        assessment = VSTEPAssessment(**output_dict)
        return True, assessment
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    # Test the schema
    schema = get_vstep_json_schema()
    print("VSTEP JSON Schema:")
    print(json.dumps(json.loads(schema), indent=2))
    
    # Test validation
    test_output = {
        "grammar": "Good control of basic structures",
        "vocabulary": "Adequate range for familiar topics", 
        "discourse": "Clear organization with basic linking",
        "total": "Solid overall performance with competent speaking ability"
    }
    
    is_valid, result = validate_vstep_output(test_output)
    print(f"\nValidation test: {'PASSED' if is_valid else 'FAILED'}")
    if is_valid:
        print("Sample assessment created successfully!")
    else:
        print(f"Error: {result}")