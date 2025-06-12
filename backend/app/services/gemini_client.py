from vertexai.generative_models import GenerativeModel, Part, Content
from typing import AsyncGenerator, List, Optional
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Gemini model interactions"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = GenerativeModel(model_name)
    
    async def generate_stream(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Gemini"""
        try:
            if system_instruction:
                model = GenerativeModel(
                    model_name=self.model._model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response_stream = model.generate_content(
                prompt,
                stream=True
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate non-streaming response from Gemini"""
        try:
            if system_instruction:
                model = GenerativeModel(
                    model_name=self.model._model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def analyze_image(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """Analyze an image with Gemini vision"""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            image_part = Part.from_data(
                mime_type="image/png",
                data=image_bytes
            )
            
            response = self.model.generate_content([prompt, image_part])
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise