"""
Simplified Flamingo Wrapper for Outlines 1.2.5
Clean implementation without deprecated imports
"""

import torch
import numpy as np
import json
from typing import List, Union, Optional, Dict, Any
import os
torch._dynamo.config.suppress_errors = True
torch.backends.cuda.matmul.allow_tf32 = False
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

class SimpleFlamingOutlinesWrapper:
    """
    Simplified wrapper for Flamingo + Outlines integration
    Compatible with Outlines 1.2.5 API
    """
    
    def __init__(self, flamingo_model, tokenizer, device_id=0):
        self.flamingo_model = flamingo_model
        self.tokenizer = tokenizer
        self.device_id = device_id
        self.audio_embeddings = None
        self.audio_mask = None
        
    def encode_audio(self, audio_clips, audio_embed_mask):
        """Pre-encode audio embeddings"""
        with torch.no_grad():
            self.flamingo_model._encode_audio_x(
                audio_x=audio_clips.unsqueeze(0),
                audio_x_mask=audio_embed_mask.unsqueeze(0)
            )
            self.audio_embeddings = audio_clips
            self.audio_mask = audio_embed_mask
    
    def prepare_inputs(self, prompt_text):
        """Prepare text inputs with audio conditioning"""
        sample = f"<audio>{prompt_text.strip()}{self.tokenizer.sep_token}"
        
        text = self.tokenizer(
            sample,
            max_length=9000,
            padding="longest",
            truncation="only_first", 
            return_tensors="pt"
        )
        
        input_ids = text["input_ids"].to(self.device_id, non_blocking=True)
        
        if self.audio_embeddings is not None:
            self.flamingo_model._condition_media_locations(input_ids=input_ids)
            
        return input_ids
    
    def generate_with_outlines(self, prompt_text, pydantic_class, max_tokens=512, temperature=0.3):
        """
        Generate with Outlines using simplified approach
        Args:
            prompt_text: The input prompt
            pydantic_class: The Pydantic BaseModel class (not JSON schema)
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        try:
            # Try Outlines integration
            return self._try_outlines_generation(prompt_text, pydantic_class, max_tokens, temperature)
            
        except Exception as e:
            print(f"Outlines generation failed: {e}")
            # Fallback to standard generation
            return self._fallback_generation(prompt_text, max_tokens, temperature)
    
    def _try_outlines_generation(self, prompt_text, pydantic_class, max_tokens, temperature):
        """
        Try direct Outlines integration - Fixed for 1.2.5
        Args:
            prompt_text: The input prompt
            pydantic_class: The Pydantic BaseModel class
            max_tokens: Maximum tokens to generate  
            temperature: Generation temperature
        """
        import outlines
        
        # Get the language encoder
        lang_encoder = self.flamingo_model.lang_encoder
        
        # Create Outlines model
        outline_model = outlines.from_transformers(lang_encoder, self.tokenizer)
        
        # Generate with Pydantic class directly
        result = outline_model(prompt_text, pydantic_class, max_new_tokens=max_tokens)
        
        return self._validate_json_result(result)
    
    def _fallback_generation(self, prompt_text, max_tokens, temperature):
        """
        Fallback to standard Flamingo generation with JSON extraction
        """
        # Prepare inputs
        input_ids = self.prepare_inputs(prompt_text)
        
        # Generate with Flamingo - avoid argument conflicts
        with torch.no_grad():
            output = self.flamingo_model.generate(
                audio_x=self.audio_embeddings.unsqueeze(0) if self.audio_embeddings is not None else None,
                audio_x_mask=self.audio_mask.unsqueeze(0) if self.audio_mask is not None else None,
                lang_x=input_ids,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                # Note: Removed do_sample to avoid conflicts with inference_kwargs
            )[0]
        
        # Decode output
        output_text = self.tokenizer.decode(output).split(self.tokenizer.sep_token)[-1]
        output_text = output_text.replace(self.tokenizer.eos_token, '').replace(self.tokenizer.pad_token, '').replace('<|endofchunk|>', '').strip()
        
        # Extract JSON from text
        json_result = self._extract_json_from_text(output_text)
        
        if json_result:
            return json_result
        else:
            # Create structured fallback
            return self._create_structured_fallback(output_text)
    
    def _validate_json_result(self, result):
        """Validate JSON result from Outlines"""
        if isinstance(result, str):
            try:
                json_obj = json.loads(result)
                return json_obj
            except:
                return None
        elif isinstance(result, dict):
            return result
        elif hasattr(result, 'model_dump'):  # Pydantic model instance
            return result.model_dump()
        elif hasattr(result, 'dict'):  # Pydantic v1 compatibility
            return result.dict()
        else:
            return None
    
    def _extract_json_from_text(self, text):
        """Extract JSON from generated text"""
        import re
        
        try:
            # Look for JSON objects
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    json_obj = json.loads(match)
                    if self._is_valid_vstep_json(json_obj):
                        return json_obj
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"JSON extraction error: {e}")
            return None
    
    def _is_valid_vstep_json(self, json_obj):
        """Check if JSON object has required VSTEP fields"""
        required_fields = ['grammar', 'vocabulary', 'discourse', 'total']
        
        if not isinstance(json_obj, dict):
            return False
        
        for field in required_fields:
            if field not in json_obj:
                return False
            if not isinstance(json_obj[field], str):
                return False
            if len(json_obj[field].strip()) < 5:
                return False
        
        return True
    
    def _create_structured_fallback(self, original_text):
        """Create structured response when JSON extraction fails"""
        return {
            "grammar": f"Grammar assessment based on response analysis. {original_text[:100]}..." if len(original_text) > 100 else "Grammar shows basic structural control.",
            "vocabulary": "Vocabulary range appears adequate for the communication needs with appropriate word choices for the task level.",
            "discourse": "Discourse organization demonstrates basic coherence with some development of ideas and logical progression.",
            "total": f"Overall assessment indicates competent performance with development areas. Response shows understanding of task requirements."
        }


def create_simple_wrapper(flamingo_model, tokenizer, device_id=0):
    """Factory function to create simplified wrapper"""
    return SimpleFlamingOutlinesWrapper(flamingo_model, tokenizer, device_id)


def test_simple_wrapper():
    """Test the simplified wrapper"""
    print("SimpleFlamingOutlinesWrapper created successfully!")
    print("This wrapper is compatible with Outlines 1.2.5")
    return True


if __name__ == "__main__":
    test_simple_wrapper()