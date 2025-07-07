"""
Function Calling System
Handles parsing function calls from LLM responses and executing them.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from .call_center_functions import FUNCTION_REGISTRY

logger = logging.getLogger(__name__)

class FunctionCaller:
    """
    Class for managing function calls from the LLM.
    """
    
    def __init__(self):
        """Initialize the function caller."""
        self.registry = FUNCTION_REGISTRY
        logger.info(f"Initialized FunctionCaller with {len(self.registry)} functions")
        
    def get_available_functions(self) -> Dict[str, Any]:
        """
        Get list of available functions and their schemas.
        
        Returns:
            Dictionary of available functions with their schemas
        """
        functions_schema = {}
        for func_name, func_info in self.registry.items():
            functions_schema[func_name] = {
                "description": func_info["description"],
                "parameters": func_info["parameters"]
            }
        return functions_schema
    
    def parse_function_call(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from LLM response text.
        
        Looks for patterns like:
        - { "function_name(param1=\"value1\")" } (Incomplete JSON - missing value)
        - { "FUNCTION_CALL": [ "function_name(param1=\"value1\")" ] } (JSON array format)
        - { "function_call": { "name": "function_name", "parameters": {...} } } (Structured JSON)
        - {"function_name(param1=\"value1\")"} (Simple JSON format)  
        - FUNCTION_CALL: function_name(param1="value1", param2="value2")
        - [CALL] function_name {"param1": "value1", "param2": "value2"}
        
        Args:
            text: The text to parse for function calls
            
        Returns:
            List of parsed function calls
        """
        function_calls = []
        
        # Pattern 0a: Handle invalid JSON format: { "function_name(params)" } (missing value)
        # Only match if it's exactly this format - no colons or other structure
        incomplete_json_pattern = r'^\{\s*"(\w+\([^)]*\))"\s*\}$'
        match = re.match(incomplete_json_pattern, text.strip())
        if match:
            func_call_str = match.group(1)
            # Parse the function call string
            func_match = re.match(r'(\w+)\((.*?)\)', func_call_str)
            if func_match:
                func_name, params_str = func_match.groups()
                params = self._parse_parameters(params_str)
                function_calls.append({
                    "function": func_name,
                    "parameters": params
                })
                logger.debug(f"Parsed incomplete JSON function call: {func_name} with params: {params}")
                logger.info(f"Parsed {len(function_calls)} function calls from incomplete JSON format")
                return function_calls
        
        # Pattern 0b: New JSON format { "FUNCTION_CALL": [ "function_name(param1=\"value1\")" ] }
        try:
            # First try to parse the entire text as JSON
            if text.strip().startswith('{') and text.strip().endswith('}'):
                logger.debug(f"Attempting to parse JSON: {text.strip()}")
                json_data = json.loads(text.strip())
                logger.debug(f"Successfully parsed JSON with keys: {list(json_data.keys())}")
                
                # Format 1: { "FUNCTION_CALL": [ "function_name(params)" ] }
                if "FUNCTION_CALL" in json_data:
                    func_list = json_data["FUNCTION_CALL"]
                    if isinstance(func_list, list):
                        for func_call_str in func_list:
                            # Parse each function call string like "function_name(param1=\"value1\")"
                            match = re.match(r'(\w+)\((.*?)\)', func_call_str.strip())
                            if match:
                                func_name, params_str = match.groups()
                                params = self._parse_parameters(params_str)
                                function_calls.append({
                                    "function": func_name,
                                    "parameters": params
                                })
                                logger.debug(f"Parsed JSON format function call: {func_name} with params: {params}")
                    elif isinstance(func_list, str):
                        # Single function call as string
                        match = re.match(r'(\w+)\((.*?)\)', func_list.strip())
                        if match:
                            func_name, params_str = match.groups()
                            params = self._parse_parameters(params_str)
                            function_calls.append({
                                "function": func_name,
                                "parameters": params
                            })
                            logger.debug(f"Parsed JSON format single function call: {func_name} with params: {params}")
                
                # Format 2: { "function_call": { "name": "function_name", "parameters": {...} } }
                elif "function_call" in json_data:
                    logger.debug(f"Found 'function_call' key in JSON data")
                    func_call = json_data["function_call"]
                    logger.debug(f"Function call data: {func_call}")
                    logger.debug(f"Function call type: {type(func_call)}")
                    if isinstance(func_call, dict):
                        func_name = func_call.get("name")
                        func_params = func_call.get("parameters", {})
                        
                        if func_name:
                            function_calls.append({
                                "function": func_name,
                                "parameters": func_params if isinstance(func_params, dict) else {}
                            })
                            logger.debug(f"Parsed structured JSON function call: {func_name} with params: {func_params}")
                        else:
                            logger.debug("No function name found in structured JSON")
                    elif isinstance(func_call, list):
                        # Handle array of function calls
                        for call in func_call:
                            if isinstance(call, dict):
                                func_name = call.get("name")
                                func_params = call.get("parameters", {})
                                
                                if func_name:
                                    function_calls.append({
                                        "function": func_name,
                                        "parameters": func_params if isinstance(func_params, dict) else {}
                                    })
                                    logger.debug(f"Parsed structured JSON array function call: {func_name} with params: {func_params}")
                
                # Format 3: {"function_name(params)"} - simple JSON with function call as key
                elif len(json_data) == 1:
                    for key in json_data.keys():
                        # Check if the key looks like a function call
                        match = re.match(r'(\w+)\((.*?)\)', key.strip())
                        if match:
                            func_name, params_str = match.groups()
                            params = self._parse_parameters(params_str)
                            function_calls.append({
                                "function": func_name,
                                "parameters": params
                            })
                            logger.debug(f"Parsed simple JSON function call: {func_name} with params: {params}")
                            break
                
                # If we reach here, JSON was valid but didn't match any expected format
                else:
                    logger.debug(f"Valid JSON found but doesn't match any expected function call format. Keys: {list(json_data.keys())}")
                    logger.debug(f"JSON data: {json_data}")
        except json.JSONDecodeError as e:
            logger.debug(f"Text is not valid JSON: {e}")
        except Exception as e:
            logger.debug(f"Error parsing JSON format: {e}")
        
        # If we found function calls in JSON format, return them
        if function_calls:
            logger.info(f"Parsed {len(function_calls)} function calls from JSON format")
            return function_calls
        else:
            logger.debug("No function calls found in JSON format, trying other patterns")
        
        # Pattern 1: FUNCTION_CALL: function_name(param1="value1", param2="value2")
        pattern1 = r'FUNCTION_CALL:\s*(\w+)\((.*?)\)'
        matches1 = re.findall(pattern1, text, re.DOTALL)
        
        for func_name, params_str in matches1:
            try:
                # Parse parameters
                params = self._parse_parameters(params_str)
                function_calls.append({
                    "function": func_name,
                    "parameters": params
                })
                logger.debug(f"Parsed function call: {func_name} with params: {params}")
            except Exception as e:
                logger.warning(f"Failed to parse parameters for {func_name}: {e}")
        
        # Pattern 2: [CALL] function_name {"param1": "value1", "param2": "value2"}
        pattern2 = r'\[CALL\]\s*(\w+)\s*(\{.*?\})'
        matches2 = re.findall(pattern2, text, re.DOTALL)
        
        for func_name, json_params in matches2:
            try:
                params = json.loads(json_params)
                function_calls.append({
                    "function": func_name,
                    "parameters": params
                })
                logger.debug(f"Parsed JSON function call: {func_name} with params: {params}")
            except Exception as e:
                logger.warning(f"Failed to parse JSON parameters for {func_name}: {e}")
        
        # Pattern 3: <function_call> XML-like format
        pattern3 = r'<function_call>\s*<name>(\w+)</name>\s*<parameters>(.*?)</parameters>\s*</function_call>'
        matches3 = re.findall(pattern3, text, re.DOTALL | re.IGNORECASE)
        
        for func_name, params_str in matches3:
            try:
                # Try to parse as JSON first
                params = json.loads(params_str.strip())
                function_calls.append({
                    "function": func_name,
                    "parameters": params
                })
                logger.debug(f"Parsed XML function call: {func_name} with params: {params}")
            except:
                # If JSON parsing fails, try parameter parsing
                try:
                    params = self._parse_parameters(params_str.strip())
                    function_calls.append({
                        "function": func_name,
                        "parameters": params
                    })
                    logger.debug(f"Parsed XML function call with param parsing: {func_name} with params: {params}")
                except Exception as e:
                    logger.warning(f"Failed to parse XML parameters for {func_name}: {e}")
        
        logger.info(f"Parsed {len(function_calls)} function calls from text")
        return function_calls
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """
        Parse parameters from a string like 'param1="value1", param2="value2"'.
        
        Args:
            params_str: String containing parameters
            
        Returns:
            Dictionary of parsed parameters
        """
        params = {}
        if not params_str.strip():
            return params
        
        # Try to parse as JSON first
        try:
            if params_str.strip().startswith('{'):
                return json.loads(params_str)
        except:
            pass
        
        # Parse as key=value pairs
        # Handle both quoted and unquoted values
        pattern = r'(\w+)\s*=\s*(["\']([^"\']*)["\']|([^,]+))'
        matches = re.findall(pattern, params_str)
        
        for match in matches:
            key = match[0]
            # match[2] is quoted value, match[3] is unquoted value
            value = match[2] if match[2] else match[3].strip()
            
            # Try to convert to appropriate type
            try:
                # Try number conversion
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # Keep as string
                params[key] = value
        
        return params
    
    def execute_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function with the given parameters.
        
        Args:
            function_name: Name of the function to execute
            parameters: Parameters to pass to the function
            
        Returns:
            Result of the function execution
        """
        logger.info(f"Executing function: {function_name} with parameters: {parameters}")
        
        if function_name not in self.registry:
            error_msg = f"Function '{function_name}' not found in registry"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
        
        func_info = self.registry[function_name]
        function = func_info["function"]
        
        try:
            # Validate required parameters
            required_params = [
                param_name for param_name, param_info in func_info["parameters"].items()
                if param_info.get("required", False)
            ]
            
            missing_params = [param for param in required_params if param not in parameters]
            if missing_params:
                error_msg = f"Missing required parameters: {missing_params}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "data": None
                }
            
            # Execute the function
            result = function(**parameters)
            logger.info(f"Function {function_name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error executing function {function_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
    
    def execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple function calls.
        
        Args:
            function_calls: List of function calls to execute
            
        Returns:
            List of results from each function call
        """
        results = []
        for call in function_calls:
            function_name = call["function"]
            parameters = call["parameters"]
            result = self.execute_function(function_name, parameters)
            results.append({
                "function": function_name,
                "parameters": parameters,
                "result": result
            })
        return results
    
    def parse_and_execute(self, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse function calls from text and execute them.
        
        Args:
            text: Text containing function calls
            
        Returns:
            Tuple of (execution_results, cleaned_text)
        """
        # Parse function calls
        function_calls = self.parse_function_call(text)
        
        # Execute function calls
        results = []
        if function_calls:
            results = self.execute_function_calls(function_calls)
        
        # Clean the text by removing function call syntax
        cleaned_text = self._clean_text(text)
        
        return results, cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """
        Remove function call syntax from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove incomplete JSON format first: { "function_name(params)" } (missing value)
        incomplete_json_pattern = r'\{\s*"\w+\([^)]*\)"\s*\}'
        if re.match(incomplete_json_pattern, text.strip()):
            text = ""
            return text.strip()
        
        # Remove JSON FUNCTION_CALL formats: { "FUNCTION_CALL": [ "..." ] } and {"function_name(params)"}
        try:
            if text.strip().startswith('{') and text.strip().endswith('}'):
                json_data = json.loads(text.strip())
                
                # Format 1: { "FUNCTION_CALL": [ "..." ] }
                if "FUNCTION_CALL" in json_data:
                    # This is a pure function call JSON, replace with empty string
                    text = ""
                
                # Format 2: {"function_name(params)"} - simple JSON with function call as key
                elif len(json_data) == 1:
                    for key in json_data.keys():
                        # Check if the key looks like a function call
                        if re.match(r'\w+\(.*?\)', key.strip()):
                            # This is a pure function call JSON, replace with empty string
                            text = ""
                            break
                
                # Format 3: { "function_call": { "name": "function_name", "parameters": {...} } }
                elif "function_call" in json_data:
                    # This is a structured function call JSON, replace with empty string
                    text = ""
        except (json.JSONDecodeError, Exception):
            pass
        
        # Remove FUNCTION_CALL patterns
        text = re.sub(r'FUNCTION_CALL:\s*\w+\([^)]*\)', '', text)
        
        # Remove [CALL] patterns
        text = re.sub(r'\[CALL\]\s*\w+\s*\{[^}]*\}', '', text)
        
        # Remove XML-like function call patterns
        text = re.sub(r'<function_call>.*?</function_call>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove JSON function call patterns (regex fallback)
        text = re.sub(r'\{\s*"FUNCTION_CALL"\s*:\s*\[.*?\]\s*\}', '', text, flags=re.DOTALL)
        
        # Remove simple JSON function call patterns: {"function_name(params)"}
        text = re.sub(r'\{\s*"\w+\([^)]*\)"\s*:\s*[^}]*\}', '', text, flags=re.DOTALL)
        
        # Remove incomplete JSON function call patterns: { "function_name(params)" } (missing value)
        text = re.sub(r'\{\s*"\w+\([^)]*\)"\s*\}', '', text, flags=re.DOTALL)
        
        # Remove structured JSON function call patterns: { "function_call": { "name": "...", "parameters": {...} } }
        # This regex handles nested braces for the parameters object
        text = re.sub(r'\{\s*"function_call"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text
    
    def format_function_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format function execution results for display.
        
        Args:
            results: List of function execution results
            
        Returns:
            Formatted string representation of the results
        """
        if not results:
            return ""
        
        formatted = "\n**Function Calls Executed:**\n\n"
        
        for i, result in enumerate(results, 1):
            function_name = result["function"]
            parameters = result["parameters"]
            execution_result = result["result"]
            
            formatted += f"**{i}. {function_name}**\n"
            formatted += f"   Parameters: {parameters}\n"
            
            if execution_result["success"]:
                formatted += f"   Status: ✅ Success\n"
                if execution_result["data"]:
                    formatted += f"   Result: {json.dumps(execution_result['data'], indent=2)}\n"
            else:
                formatted += f"   Status: ❌ Error\n"
                formatted += f"   Error: {execution_result['error']}\n"
            
            formatted += "\n"
        
        return formatted 