# whisper_service.py

import os
import tempfile
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class WhisperService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def transcribe_audio(self, audio_data: bytes, filename: str = "audio.webm") -> str:
        """
        Transcribe audio using OpenAI Whisper API
        
        Args:
            audio_data: Audio file bytes
            filename: Original filename (used for format detection)
        
        Returns:
            Transcribed text
        """
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            try:
                # Open the file and send to Whisper API
                with open(temp_audio_path, 'rb') as audio_file:
                    transcript = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"  # You can make this dynamic or auto-detect
                    )
                
                return transcript.text
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")
            raise Exception(f"Failed to transcribe audio: {str(e)}")

# Singleton instance
whisper_service = WhisperService()