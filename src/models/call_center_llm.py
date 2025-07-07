"""
Call Center LLM Manager
Enhanced LLM manager for call center chatbot with function calling capabilities.
"""

import os
import sys
import json
import logging
import time
import psutil
from typing import List, Optional, Tuple, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from transformers.pipelines import pipeline
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from src.utils.config import (
    DEFAULT_LLM_MODEL,
    CACHE_DIR
)
from src.functions.function_caller import FunctionCaller

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Call center system prompt
CALL_CENTER_SYSTEM_PROMPT = """You are an AI assistant for a telecommunications call center. Your role is to help customers with their account inquiries, billing questions, package changes, and technical support issues.

You have access to several functions that you can call to help customers:
1. get_customer_info(customer_id) - Get customer account information
2. get_available_packages() - Show available service packages
3. get_package_details(package_name) - Get details about a specific package
4. initiate_package_change(customer_id, new_package, effective_date) - Change customer's package
5. check_billing_status(customer_id) - Check billing and payment information
6. process_payment(customer_id, amount, payment_method) - Process a payment
7. get_usage_summary(customer_id, period) - Get usage information
8. create_support_ticket(customer_id, issue_type, description, priority) - Create a support ticket

If a function call is needed, respond **ONLY** with the following JSON format. **DO NOT** include any other text, explanations, or pleasantries outside of this JSON structure.
{
  "function_call": {
    "name": "function_name",
    "parameters": {
      "parameter1": "value1",
      "parameter2": "value2"
    }
  }
}

If **no function call is needed**, respond with customer's language reply instead.

Assume customer_id is either provided or must be asked from the customer if needed.

Example:

Customer: Can you tell me about my account?
Assistant:
{
  "function_call": {
    "name": "get_customer_info",
    "parameters": {
      "customer_id": "123456"
    }
  }
}

"""

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.call_count = 0
        logger.debug(f"Initialized CustomStoppingCriteria with stop strings: {stop_strings}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.call_count += 1
        
        # Decode the generated text
        generated = self.tokenizer.decode(input_ids[0])
        self.generated_text = generated
        # Log every 10 calls to avoid spam
        if self.call_count % 10 == 0:
            logger.debug(f"Stopping criteria check #{self.call_count}: Current generated length: {len(generated)} chars")
        
        # Check if any stop string appears in the generated text
        for stop_string in self.stop_strings:
            if stop_string in generated:
                logger.info(f"Stopping criteria triggered by: '{stop_string}' at call #{self.call_count}")
                return True
        
        return False

class CallCenterLLMManager:
    """Class for managing the call center LLM and generating responses with function calling."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        system_prompt: str = CALL_CENTER_SYSTEM_PROMPT,
        cache_dir: str = CACHE_DIR,
        device: Optional[str] = None,
        load_in_bits: Optional[int] = None
    ):
        """Initialize the call center LLM manager.
        
        Args:
            model_name: Name or path of the model to use
            system_prompt: System prompt for call center operations
            cache_dir: Directory to cache model weights
            device: Device to run the model on (auto-detected if None)
            load_in_bits: Load model in specified bits (4 or 8) for quantization
        """
        logger.info(f"Initializing CallCenterLLMManager with:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Cache dir: {cache_dir}")
        logger.info(f"  - System prompt length: {len(system_prompt)} chars")
        if load_in_bits:
            logger.info(f"  - Quantization: {load_in_bits}-bit")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        self.load_in_bits = load_in_bits
        
        # Initialize function caller
        self.function_caller = FunctionCaller()
        logger.info(f"Initialized function caller with {len(self.function_caller.registry)} functions")
        
        # Log system information
        self._log_system_info()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"Cache directory created/verified: {cache_dir}")
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"Auto-detected device: {self.device}")
        else:
            self.device = device
            logger.info(f"Using specified device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Initialize the model
        self._initialize_model()
    
    def _log_system_info(self):
        """Log system information for debugging."""
        logger.info(f"System Information:")
        logger.info(f"  - Python version: {sys.version}")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - Available CPU cores: {psutil.cpu_count()}")
        logger.info(f"  - Available RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
        logger.info(f"  - Available disk space: {psutil.disk_usage('/').free / 1e9:.2f} GB")
        
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        start_time = time.time()
        logger.info(f"Starting model initialization: {self.model_name} on {self.device}")
        
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                logger.warning("HUGGINGFACE_TOKEN environment variable not found. Trying HUGGINGFACEHUB_API_TOKEN.")
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
            if not hf_token:
                logger.error("Hugging Face token not found.")
                raise ValueError("Hugging Face token not set.")
            logger.debug("Hugging Face token found.")

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA specified but not available. Falling back to CPU.")
                self.device = "cpu"
            elif self.device == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                logger.warning("MPS specified but not available/built. Falling back to CPU.")
                self.device = "cpu"
            logger.info(f"Effective device for model and pipeline: {self.device}")

            # Load tokenizer
            tokenizer_start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=hf_token
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer pad_token set to eos_token as it was None.")
            tokenizer_time = time.time() - tokenizer_start_time
            logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds. Vocab size: {self.tokenizer.vocab_size}")

            if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                ModelClass = AutoModelForSeq2SeqLM
                pipeline_task = "text2text-generation"
                logger.info(f"Using AutoModelForSeq2SeqLM for {self.model_name}. Pipeline task: {pipeline_task}")
            else:
                ModelClass = AutoModelForCausalLM
                pipeline_task = "text-generation"
                logger.info(f"Using AutoModelForCausalLM for {self.model_name}. Pipeline task: {pipeline_task}")
            
            # Prepare model loading arguments
            model_load_args = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "token": hf_token,
                "torch_dtype": torch.float16 if (self.device == "cuda" or self.device == "mps") else torch.float32,
                "low_cpu_mem_usage": True if self.device != "cpu" else False
            }

            if self.load_in_bits == 8:
                model_load_args["load_in_8bit"] = True
                logger.info("Attempting to load model with 8-bit quantization.")
            elif self.load_in_bits == 4:
                model_load_args["load_in_4bit"] = True
                logger.info("Attempting to load model with 4-bit quantization.")

            if self.device == "cuda" or self.device == "mps":
                model_load_args["device_map"] = "auto" 
            
            logger.info(f"Loading model {self.model_name} with args: {model_load_args}")
            model_start_time = time.time()
            self.model = ModelClass.from_pretrained(self.model_name, **model_load_args)
            model_time = time.time() - model_start_time
            logger.info(f"Model loaded in {model_time:.2f} seconds.")

            if self.device == "cpu" and (not hasattr(self.model, 'hf_device_map') or not self.model.hf_device_map):
                logger.info(f"Ensuring model is on CPU device: {self.device}")
                self.model.to(self.device)

            logger.info(f"Creating Hugging Face pipeline with task: {pipeline_task}")
            pipeline_start_time = time.time()
            
            if model_load_args.get("device_map") == "auto":
                pipeline_device_arg = None
                logger.info("Model was loaded with device_map='auto'. Pipeline 'device' argument set to None.")
            elif self.device == "cpu":
                pipeline_device_arg = None
                logger.info("Model on CPU. Pipeline 'device' argument set to None.")
            else:
                pipeline_device_arg = self.device
                logger.info(f"Pipeline 'device' argument set to '{self.device}'.")
                
            self.pipe = pipeline(
                pipeline_task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device_arg
            )
            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline created in {pipeline_time:.2f} seconds.")

            total_time = time.time() - start_time
            logger.info(f"CallCenterLLMManager initialized successfully in {total_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}", exc_info=True)
            raise
    
    def _format_prompt(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt for the call center model.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted prompt string
        """
        logger.debug(f"Formatting prompt for call center model")
        logger.debug(f"Query length: {len(query)} chars")
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conversation_context += f"{role.title()}: {content}\n"
        
        # Format based on model type
        if "mistral" in self.model_name.lower():
            # Mistral-style prompt
            formatted_prompt = f"<s>[INST] {self.system_prompt}\n\n{conversation_context}Customer: {query} [/INST]"
            logger.debug("Using Mistral-style prompt formatting")
        elif "llama" in self.model_name.lower():
            formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{conversation_context}Customer: {query} [/INST]"
            logger.debug("Using LLaMA-style prompt formatting")
        elif "t5" in self.model_name.lower():
            formatted_prompt = f"{conversation_context}Customer: {query}\n\nAgent:"
            logger.debug("Using T5-style prompt formatting")
        else:
            # Generic prompt
            formatted_prompt = f"System: {self.system_prompt}\n\n{conversation_context}Customer: {query}\n\nAgent:"
            logger.debug("Using generic prompt formatting")
        
        logger.debug(f"Final prompt length: {len(formatted_prompt)} chars")
        return formatted_prompt
    
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a response to a query and execute any function calls.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            max_length: Maximum new tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (response_text, function_call_results)
        """
        generation_start_time = time.time()
        logger.info(f"Starting call center response generation")
        
        # Format the prompt
        prompt_start = time.time()
        prompt = self._format_prompt(query, conversation_history)
        prompt_time = time.time() - prompt_start

        logger.info(f"=== CALL CENTER LLM REQUEST ===")
        logger.info(f"QUERY: {query}")
        logger.info(f"PROMPT LENGTH: {len(prompt)} chars")
        logger.info(f"GENERATION PARAMETERS:")
        logger.info(f"  - MAX_NEW_TOKENS: {max_length}")
        logger.info(f"  - TEMPERATURE: {temperature}")
        logger.info(f"  - TOP_P: {top_p}")
        logger.info(f"  - MODEL: {self.model_name}")
        logger.info(f"  - DEVICE: {self.device}")
        logger.info(f"==============================")
        
        try:
            # Define stopping criteria - more conservative to avoid premature stopping
            stop_strings = ["<|endoftext|>", "</s>"]  # Only use explicit end tokens
            stopping_criteria = StoppingCriteriaList([
                CustomStoppingCriteria(stop_strings, self.tokenizer)
            ])
            
            # Generate response
            generation_actual_start = time.time()
            logger.info(f"Starting actual text generation...")
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_length, 
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
            
            generation_actual_time = time.time() - generation_actual_start
            logger.info(f"Text generation completed in {generation_actual_time:.2f}s")
            
            # Handle pipeline output
            generated_text = ""
            if outputs:
                if isinstance(outputs, list) and len(outputs) > 0:
                    first_output = outputs[0]
                    if isinstance(first_output, dict) and "generated_text" in first_output:
                        generated_text = first_output["generated_text"]
                elif hasattr(outputs, '__iter__') and not isinstance(outputs, (str, dict)):
                    outputs_list = list(outputs)
                    if outputs_list and isinstance(outputs_list[0], dict):
                        generated_text = outputs_list[0].get("generated_text", "")
                elif isinstance(outputs, dict) and "generated_text" in outputs:
                    generated_text = outputs["generated_text"]
            
            logger.info(f"=== GENERATION OUTPUT ===")
            logger.info(f"RAW OUTPUT LENGTH: {len(generated_text)} chars")
            logger.debug(f"RAW OUTPUT: {generated_text}")
            logger.info(f"========================")

            # Extract response from generated text
            extraction_start = time.time()
            response = self._extract_response(str(generated_text), prompt)
            extraction_time = time.time() - extraction_start
            
            # Parse and execute function calls
            function_results, cleaned_response = self.function_caller.parse_and_execute(response)
            
            # If function calls were executed and response is empty/minimal, generate a natural response
            if function_results and (not cleaned_response or len(cleaned_response.strip()) < 10):
                logger.info("Function calls executed but response is empty. Generating natural language response...")
                
                # Create a follow-up prompt with function results
                follow_up_response = self._generate_function_response(query, function_results, conversation_history)
                if follow_up_response:
                    cleaned_response = follow_up_response
            
            total_generation_time = time.time() - generation_start_time
            
            logger.info(f"=== FINAL CALL CENTER RESPONSE ===")
            logger.info(f"EXTRACTION TIME: {extraction_time:.4f}s")
            logger.info(f"FUNCTION CALLS FOUND: {len(function_results)}")
            logger.info(f"TOTAL GENERATION TIME: {total_generation_time:.2f}s")
            logger.info(f"CLEANED RESPONSE: {cleaned_response}")
            logger.info(f"================================")
            
            return cleaned_response, function_results

        except Exception as e:
            logger.error(f"Error generating call center response: {e}", exc_info=True)
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact technical support.", []
    
    def _extract_response(self, generated_text: str, original_prompt: str) -> str:
        """Extract the assistant's response from the generated text.
        
        Args:
            generated_text: Full generated text from the model
            original_prompt: The original prompt sent to the model
            
        Returns:
            Extracted response text
        """
        logger.debug(f"Extracting response from generated text (length: {len(generated_text)})")
        logger.debug(f"Generated text: {generated_text}")
        
        # For Mistral/Llama models, the response comes after [/INST]
        if "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
            inst_token = "[/INST]"
            logger.debug(f"Using Mistral/Llama extraction with token: {inst_token}")
            
            # Find [/INST] in the generated text (not in the original prompt)
            inst_index = generated_text.rfind(inst_token)
            
            if inst_index != -1:
                # Extract everything after [/INST]
                response_start = inst_index + len(inst_token)
                response = generated_text[response_start:].strip()
                logger.debug(f"Found [/INST] at index {inst_index}, extracted response: '{response}'")
            else:
                # No [/INST] found, try to remove the original prompt
                if generated_text.startswith(original_prompt):
                    response = generated_text[len(original_prompt):].strip()
                    logger.debug(f"No [/INST] found, removed prompt by length, response: '{response}'")
                else:
                    response = generated_text.strip()
                    logger.debug(f"No [/INST] found and text doesn't start with prompt, using full text: '{response}'")
        
        elif "t5" in self.model_name.lower():
            # T5 models usually output only the response
            response = generated_text.strip()
            logger.debug(f"T5 model, using full text: '{response}'")
        
        else:
            # Generic: try to remove the prompt
            logger.debug(f"Using generic extraction method")
            if generated_text.startswith(original_prompt):
                response = generated_text[len(original_prompt):].strip()
                logger.debug(f"Removed prompt by length, response: '{response}'")
            elif len(generated_text) > len(original_prompt) and generated_text.startswith(original_prompt[:100]):
                response = generated_text[len(original_prompt):].strip()
                logger.debug(f"Removed prompt by partial match, response: '{response}'")
            else:
                response = generated_text.strip()
                logger.debug(f"Could not remove prompt, using full text: '{response}'")
        
        # Clean up any remaining stop strings
        original_response = response
        stop_strings = ["Customer:", "Human:", "<|endoftext|>", "</s>", "User:"]
        for stop_string in stop_strings:
            if stop_string in response:
                response = response.split(stop_string)[0].strip()
                logger.debug(f"Cleaned stop string '{stop_string}': '{original_response}' -> '{response}'")
        
        logger.debug(f"Final extracted response: '{response}' (length: {len(response)})")
        return response
    
    def _generate_function_response(
        self,
        original_query: str,
        function_results: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a natural language response using function call results.
        
        Args:
            original_query: The customer's original query
            function_results: Results from executed functions
            conversation_history: Previous conversation messages
            
        Returns:
            Natural language response incorporating the function results
        """
        logger.info("Generating natural language response from function results")
        
        # Build function results context
        function_context = "Based on the following information retrieved from our systems:\n\n"
        
        for result in function_results:
            func_name = result["function"]
            func_result = result["result"]
            
            if func_result["success"] and func_result["data"]:
                function_context += f"• {func_name}: {json.dumps(func_result['data'], indent=2)}\n"
            elif not func_result["success"]:
                function_context += f"• {func_name}: Error - {func_result['error']}\n"
        
        function_context += "\nPlease provide a helpful, natural response to the customer."
        
        # Create new prompt for response generation
        response_prompt = f"""You are a helpful call center agent. A customer asked: "{original_query}"

{function_context}

Respond naturally and helpfully to the customer using this information. Do not include any function calls in your response - just provide a clear, conversational answer.

Respond in customer's language."""
        
        logger.debug(f"Function response prompt: {response_prompt}")
        
        try:
            # Generate natural language response
            outputs = self.pipe(
                response_prompt,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract response
            response_text = ""
            if outputs:
                if isinstance(outputs, list) and len(outputs) > 0:
                    first_output = outputs[0]
                    if isinstance(first_output, dict) and "generated_text" in first_output:
                        response_text = first_output["generated_text"]
                elif isinstance(outputs, dict) and "generated_text" in outputs:
                    response_text = outputs["generated_text"]
            
            # Clean up the response - remove the prompt
            if response_text and len(response_text) > len(response_prompt):
                response_text = response_text[len(response_prompt):].strip()
            
            # Clean any remaining artifacts
            response_text = response_text.replace("Assistant:", "").replace("Agent:", "").strip()
            
            logger.info(f"Generated function response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating function response: {e}")
            # Fallback: create a simple response from function results
            return self._create_simple_function_response(function_results)
    
    def _create_simple_function_response(self, function_results: List[Dict[str, Any]]) -> str:
        """Create a simple response from function results as fallback.
        
        Args:
            function_results: Results from executed functions
            
        Returns:
            Simple formatted response
        """
        if not function_results:
            return "I've processed your request."
        
        response = "Here's what I found:\n\n"
        
        for result in function_results:
            func_name = result["function"]
            func_result = result["result"]
            
            if func_result["success"] and func_result["data"]:
                if func_name == "get_customer_info":
                    data = func_result["data"]
                    response += f"Customer: {data.get('name', 'N/A')}\n"
                    response += f"Account Status: {data.get('account_status', 'N/A')}\n"
                    response += f"Current Package: {data.get('current_package', 'N/A')}\n"
                    response += f"Balance: ${data.get('account_balance', 0):.2f}\n\n"
                
                elif func_name == "get_available_packages":
                    packages = func_result["data"]
                    response += "Available packages:\n"
                    for pkg in packages:
                        response += f"• {pkg['name']}: ${pkg['price']}/month - {pkg['description']}\n"
                    response += "\n"
                
                else:
                    response += f"{func_name}: {json.dumps(func_result['data'], indent=2)}\n\n"
            
            elif not func_result["success"]:
                response += f"Error with {func_name}: {func_result['error']}\n\n"
        
        response += "Is there anything specific you'd like to know more about?"
        return response
    
    def get_available_functions(self) -> Dict[str, Any]:
        """Get information about available functions.
        
        Returns:
            Dictionary of available functions and their descriptions
        """
        return self.function_caller.get_available_functions() 