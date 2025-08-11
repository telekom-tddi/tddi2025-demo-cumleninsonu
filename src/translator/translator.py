"""
Bidirectional English-Turkish Translator
A tool for translating text between English and Turkish using Hugging Face models.
"""

import os
import sys
import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Apply performance optimization for compatible GPUs
# if torch.cuda.is_available():
    # torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Translator:
    """Manager for handling translations between English and Turkish."""
    
    def __init__(
        self,
        model_en_tr: str = "Helsinki-NLP/opus-mt-tc-big-en-tr",
        model_tr_en: str = "Helsinki-NLP/opus-mt-tc-big-tr-en",
        cache_dir: str = "./cache",
        device: str = None
    ):
        """Initialize the translation manager."""
        logger.info("Initializing Translator...")
        
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info("Initializing English to Turkish model...")
        self.pipe_en_to_tr = self._initialize_pipeline(
            model_en_tr,
            "translation_en_to_tr"
        )
        
        logger.info("Initializing Turkish to English model...")
        self.pipe_tr_to_en = self._initialize_pipeline(
            model_tr_en,
            "translation_tr_to_en"
        )
    def _initialize_pipeline(self, model_name: str, task: str):
        """Initialize a translation pipeline."""
        start_time = time.time()
        logger.info(f"Loading model: {model_name} for task: {task}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model.to(self.device)
            
            pipe = pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            total_time = time.time() - start_time
            logger.info(f"Model '{model_name}' loaded successfully in {total_time:.2f} seconds")
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def _translate(self, pipe: pipeline, query: str) -> str:
        """Helper function to perform translation."""
        try:
            outputs = pipe(query, max_length=512)
            
            translated_text = ""
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                translated_text = outputs[0].get("translation_text", "")
            return translated_text
        except Exception as e:
            logger.error(f"Error generating translation: {e}")
            return "I'm sorry, I encountered an error during translation."

    def translate_en_to_tr(self, text: str) -> str:
        """Translate English text to Turkish."""
        return self._translate(self.pipe_en_to_tr, text)

    def translate_tr_to_en(self, text: str) -> str:
        """Translate Turkish text to English."""
        return self._translate(self.pipe_tr_to_en, text)
