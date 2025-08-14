"""
Call Center LLM Manager with Google GenAI API
API-based LLM manager for call center chatbot with function calling capabilities.
Uses Google GenAI API for Gemma-3-27B-IT without local CUDA requirements.
"""

import os
import json
import logging
import time
from typing import List, Optional, Tuple, Dict, Any

from google import genai
from src.functions.function_caller import FunctionCaller

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Gemma-3-27B-IT model identifier for Google GenAI API
GEMMA_MODEL_NAME = "gemma-3-27b-it"

# Call center system prompt optimized for Gemma
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

If **no function call is needed**, respond with natural reply instead.

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

If the user asks general questions or FAQs, answer naturally. Only call functions when user-specific data or action is required. Answer in customer's language."""

class CallCenterLLMManagerWithAPI:
    """Class for managing the call center LLM with Google GenAI API calls to Gemma-3-27B-IT."""
    
    def __init__(
        self,
        model_name: str = GEMMA_MODEL_NAME,
        system_prompt: str = CALL_CENTER_SYSTEM_PROMPT,
        api_key: Optional[str] = None
    ):
        """Initialize the call center LLM manager with Google GenAI API.
        
        Args:
            model_name: Name of the model to use via API
            system_prompt: System prompt for call center operations
            api_key: Google GenAI API key (if None, gets from environment)
        """
        logger.info(f"Initializing CallCenterLLMManagerWithAPI with:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - System prompt length: {len(system_prompt)} chars")

        self.model_name = model_name
        self.system_prompt = system_prompt
        
        # Initialize function caller
        self.function_caller = FunctionCaller()
        logger.info(f"Initialized function caller with {len(self.function_caller.registry)} functions")
        
        # Get API key
        if api_key is None:
            api_key = self._get_api_key()
        self.api_key = api_key
        
        # Initialize Google GenAI client
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Google GenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            raise
        
        # Test API connection
        self._test_api_connection()
    
    def _get_api_key(self) -> str:
        """Get Google GenAI API key from environment variables."""
        api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_GENAI_API_KEY environment variable not found. Trying GOOGLE_API_KEY.")
            api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            logger.error("Google GenAI API key not found in environment variables.")
            raise ValueError("Google GenAI API key not set. Please set GOOGLE_GENAI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        logger.debug("Google GenAI API key found.")
        return api_key
    
    def _test_api_connection(self):
        """Test the API connection with a simple request."""
        logger.info("Testing Google GenAI API connection...")
        
        try:
            test_response = self.client.models.generate_content(
                model=self.model_name,
                contents="Hello, test connection."
            )
            
            if test_response and hasattr(test_response, 'text'):
                logger.info("Google GenAI API connection test successful!")
                logger.debug(f"Test response: {test_response.text[:100]}...")
            else:
                logger.warning("API connection test returned unexpected response format")
                
        except Exception as e:
            logger.error(f"Google GenAI API connection test failed: {e}")
            logger.warning("Continuing anyway - API might work for actual requests.")
    
    def _format_prompt(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt for Gemma model.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted prompt string
        """
        logger.debug(f"Formatting prompt for Gemma API")
        logger.debug(f"Query length: {len(query)} chars")
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Keep last 5 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conversation_context += f"{role.title()}: {content}\n"
        
        # Format prompt
        formatted_prompt = f"{self.system_prompt}\n\n{conversation_context}Customer: {query}"
        
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
        """Generate a response to a query using Google GenAI API and execute any function calls.
        
        Args:
            query: User query
            conversation_history: Previous conversation messages
            max_length: Maximum new tokens to generate (passed to API config if supported)
            temperature: Temperature for generation (passed to API config if supported)
            top_p: Top-p sampling parameter (passed to API config if supported)
            
        Returns:
            Tuple of (response_text, function_call_results)
        """
        generation_start_time = time.time()
        logger.info(f"Starting call center response generation with Google GenAI API")
        
        # Format the prompt
        prompt_start = time.time()
        prompt = self._format_prompt(query, conversation_history)
        prompt_time = time.time() - prompt_start

        logger.info(f"=== GOOGLE GENAI API CALL CENTER LLM REQUEST ===")
        logger.info(f"QUERY: {query}")
        logger.info(f"PROMPT LENGTH: {len(prompt)} chars")
        logger.info(f"GENERATION PARAMETERS:")
        logger.info(f"  - MAX_LENGTH: {max_length}")
        logger.info(f"  - TEMPERATURE: {temperature}")
        logger.info(f"  - TOP_P: {top_p}")
        logger.info(f"  - MODEL: {self.model_name}")
        logger.info(f"============================================")
        
        try:
            # Generate response via Google GenAI API
            api_start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            api_time = time.time() - api_start_time
            
            logger.info(f"Google GenAI API call completed in {api_time:.2f}s")
            
            # Extract generated text
            generated_text = ""
            if response and hasattr(response, 'text'):
                generated_text = response.text
            else:
                logger.error("No text found in API response")
                generated_text = ""
            
            logger.info(f"=== GOOGLE GENAI API GENERATION OUTPUT ===")
            logger.info(f"RAW OUTPUT LENGTH: {len(generated_text)} chars")
            logger.debug(f"RAW OUTPUT: {generated_text}")
            logger.info(f"========================================")

            # Parse and execute function calls
            extraction_start = time.time()
            function_results, cleaned_response = self.function_caller.parse_and_execute(generated_text)
            extraction_time = time.time() - extraction_start
            
            # If function calls were executed and response is empty/minimal, generate a natural response
            if function_results:
                logger.info(f"Function calls executed: {function_results}")
                # Create a follow-up response with function results
                follow_up_response = self._generate_function_response(query, function_results, conversation_history)
                if follow_up_response:
                    cleaned_response = follow_up_response
            
            total_generation_time = time.time() - generation_start_time
            
            logger.info(f"=== FINAL GOOGLE GENAI API CALL CENTER RESPONSE ===")
            logger.info(f"EXTRACTION TIME: {extraction_time:.4f}s")
            logger.info(f"FUNCTION CALLS FOUND: {len(function_results)}")
            logger.info(f"TOTAL GENERATION TIME: {total_generation_time:.2f}s")
            logger.info(f"CLEANED RESPONSE: {cleaned_response}")
            logger.info(f"GENTEXT: {generated_text}")
            logger.info(f"FUNCRESULTS: {function_results}")
            logger.info(f"===============================================")
            
            return cleaned_response, function_results

        except Exception as e:
            logger.error(f"Error generating call center response with Google GenAI API: {e}", exc_info=True)
            return "I apologize, but I'm experiencing technical difficulties. Please try again or contact technical support.", []
    
    def _generate_function_response(
        self,
        original_query: str,
        function_results: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a natural language response using function call results with Google GenAI API.
        
        Args:
            original_query: The customer's original query
            function_results: Results from executed functions
            conversation_history: Previous conversation messages
            
        Returns:
            Natural language response incorporating the function results
        """
        logger.info("Generating natural language response from function results using Google GenAI API")
        
        # Build function results context
        function_context_parts = []
        for result in function_results:
            func_name = result["function"]
            func_result = result["result"]
            
            if func_result["success"] and func_result["data"]:
                function_context_parts.append(f"• {func_name}: {json.dumps(func_result['data'], indent=2)}")
            elif not func_result["success"]:
                function_context_parts.append(f"• {func_name}: Error - {func_result['error']}")
        
        function_context = "\n".join(function_context_parts)
        
        # Create new prompt for response generation
        response_prompt = f"""You are a helpful and empathetic call center AI assistant. Your goal is to provide a clear and concise answer to the customer's query based *only* on the information provided below.

## Customer's Original Question:
"{original_query}"

## Information from our Systems:
{function_context}

## Your Task:
1. Synthesize the information above to answer the customer's question.
2. Respond in a natural, conversational, and helpful tone.
3. **Do not** mention that you are an AI or that you retrieved information from a system.
4. **Do not** output any function calls or JSON.
5. If the system returned an error or no data, politely inform the customer that you couldn't retrieve the information and suggest what they can do next (e.g., try again later, or check the information they provided).

Please provide your response now:"""
        
        logger.debug(f"Google GenAI API function response prompt: {response_prompt}")
        
        try:
            # Generate natural language response via API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=response_prompt
            )
            
            # Extract response text
            response_text = ""
            if response and hasattr(response, 'text'):
                response_text = response.text.strip()
            
            logger.info(f"Generated Google GenAI API function response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating function response with Google GenAI API: {e}")
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