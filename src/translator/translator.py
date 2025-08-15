"""
Bidirectional English-Turkish Translator
A tool for translating text between English and Turkish using Hugging Face models.
"""

import os
import sys
import logging
import time
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
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

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences using regex patterns."""
        # More robust pattern to split sentences, handling various cases
        # Split on sentence endings followed by whitespace and capital letter or bullet points
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z*•\-])|(?<=[.!?])\s*\n\s*(?=[A-Z*•\-])'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean up empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle cases where bullet points or list items should stay together
        cleaned_sentences = []
        current_sentence = ""
        for sentence in sentences:
            # If sentence starts with bullet point or dash, combine with previous
            if sentence.startswith(('*', '•', '-', '–', '—')) and current_sentence:
                current_sentence += " " + sentence
            else:
                if current_sentence:
                    cleaned_sentences.append(current_sentence)
                current_sentence = sentence
        
        # Add the last sentence
        if current_sentence:
            cleaned_sentences.append(current_sentence)
            
        return cleaned_sentences

    def _chunk_sentences(self, sentences: list, chunk_size: int = 3) -> list:
        """Group sentences into chunks of specified size."""
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _translate_with_chunking(self, pipe: pipeline, query: str, chunk_size: int = 3) -> str:
        """Translate text by splitting into smaller chunks."""
        try:
            # Check if text is long enough to warrant chunking
            word_count = len(query.split())
            
            # If text is short enough, translate directly
            if word_count <= 50:  # Threshold for using chunking
                return self._translate_single(pipe, query)
            
            logger.info(f"Text is long ({word_count} words), using chunking approach with {chunk_size} sentences per chunk")
            
            # Split into sentences and then into chunks
            sentences = self._split_into_sentences(query)
            chunks = self._chunk_sentences(sentences, chunk_size)
            
            logger.info(f"Split text into {len(sentences)} sentences and {len(chunks)} chunks")
            
            # Translate each chunk
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Translating chunk {i+1}/{len(chunks)}")
                translated_chunk = self._translate_single(pipe, chunk)
                translated_chunks.append(translated_chunk)
            
            # Combine translated chunks
            final_translation = ' '.join(translated_chunks)
            
            logger.info("Successfully completed chunked translation")
            return final_translation
            
        except Exception as e:
            logger.error(f"Error in chunked translation: {e}")
            return "I'm sorry, I encountered an error during translation."

    def _translate_single(self, pipe: pipeline, query: str) -> str:
        """Helper function to perform single chunk translation."""
        try:
            # For single chunks, use more conservative max_length
            input_length = len(query.split())
            # Since we're working with chunks, we can be more generous with max_length
            estimated_output_length = int(input_length * 1.5) + 50
            max_length = max(256, min(1024, estimated_output_length))
            
            logger.debug(f"Single chunk - Input: {input_length} words, max_length: {max_length}")
            
            outputs = pipe(query, max_length=max_length, do_sample=False)
            
            translated_text = ""
            if outputs and isinstance(outputs, list) and len(outputs) > 0:
                translated_text = outputs[0].get("translation_text", "")
            return translated_text
        except Exception as e:
            logger.error(f"Error in single chunk translation: {e}")
            return "I'm sorry, I encountered an error during translation."

    def translate_en_to_tr(self, text: str, chunk_size: int = 3) -> str:
        """
        Translate English text to Turkish using chunking for longer texts.
        
        Args:
            text: English text to translate
            chunk_size: Number of sentences per chunk (default: 3)
        
        Returns:
            Translated Turkish text
        """
        try:
            detected_lang = detect(text)
            if detected_lang == 'tr':
                return text
        except:
            pass  # Dil tespit edilemezse çeviriye devam et
    
        return self._translate_with_chunking(self.pipe_en_to_tr, text, chunk_size)
