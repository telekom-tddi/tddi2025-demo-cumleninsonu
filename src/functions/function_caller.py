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
        - { "function_call": { "name": "function_name", "parameters": {...} } } (Structured JSON)
        - { "FUNCTION_CALL": [ "function_name(param1=\"value1\")" ] } (JSON array format)
        - { "function_name(param1=\"value1\")" } (Incomplete JSON - missing value)
        - {"function_name(param1=\"value1\")"} (Simple JSON format)  
        - FUNCTION_CALL: function_name(param1="value1", param2="value2")
        - [CALL] function_name {"param1": "value1", "param2": "value2"}
        
        Args:
            text: The text to parse for function calls
            
        Returns:
            List of parsed function calls
        """
        function_calls = []
        original_text = text
        
        # First, extract potential JSON blocks from text (handling markdown code blocks)
        json_candidates = self._extract_json_blocks(text)
        
        # Try to parse each JSON candidate
        for json_text in json_candidates:
            logger.debug(f"Trying to parse JSON candidate: {json_text}")
            
            try:
                json_data = json.loads(json_text)
                logger.debug(f"Successfully parsed JSON with keys: {list(json_data.keys())}")
                
                # Format 1: { "function_call": { "name": "function_name", "parameters": {...} } }
                if "function_call" in json_data:
                    logger.debug(f"Found 'function_call' key in JSON data")
                    func_call = json_data["function_call"]
                    logger.debug(f"Function call data: {func_call}")
                    
                    if isinstance(func_call, dict):
                        func_name = func_call.get("name")
                        func_params = func_call.get("parameters", {})
                        
                        if func_name:
                            function_calls.append({
                                "function": func_name,
                                "parameters": func_params if isinstance(func_params, dict) else {}
                            })
                            logger.info(f"✅ Parsed structured JSON function call: {func_name} with params: {func_params}")
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
                                    logger.info(f"✅ Parsed structured JSON array function call: {func_name} with params: {func_params}")
                
                # Format 2: { "FUNCTION_CALL": [ "function_name(params)" ] }
                elif "FUNCTION_CALL" in json_data:
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
                                logger.info(f"✅ Parsed JSON format function call: {func_name} with params: {params}")
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
                            logger.info(f"✅ Parsed JSON format single function call: {func_name} with params: {params}")
                
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
                            logger.info(f"✅ Parsed simple JSON function call: {func_name} with params: {params}")
                            break
                
                # If we found function calls, break out of the loop
                if function_calls:
                    break
                    
            except json.JSONDecodeError as e:
                logger.debug(f"JSON candidate is not valid JSON: {e}")
                continue
            except Exception as e:
                logger.debug(f"Error parsing JSON candidate: {e}")
                continue
        
        # If we found function calls in JSON format, return them
        if function_calls:
            logger.info(f"🎉 Successfully parsed {len(function_calls)} function calls from JSON format")
            return function_calls
        
        # Fallback to other patterns if JSON parsing failed
        logger.debug("No function calls found in JSON format, trying other patterns")
        
        # Pattern: Handle incomplete JSON format: { "function_name(params)" } (missing value)
        incomplete_json_pattern = r'\{\s*"(\w+\([^)]*\))"\s*\}'
        matches = re.findall(incomplete_json_pattern, text)
        for func_call_str in matches:
            func_match = re.match(r'(\w+)\((.*?)\)', func_call_str)
            if func_match:
                func_name, params_str = func_match.groups()
                params = self._parse_parameters(params_str)
                function_calls.append({
                    "function": func_name,
                    "parameters": params
                })
                logger.info(f"✅ Parsed incomplete JSON function call: {func_name} with params: {params}")
        
        if function_calls:
            return function_calls
        
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
                logger.info(f"✅ Parsed function call: {func_name} with params: {params}")
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
                logger.info(f"✅ Parsed JSON function call: {func_name} with params: {params}")
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
                logger.info(f"✅ Parsed XML function call: {func_name} with params: {params}")
            except:
                # If JSON parsing fails, try parameter parsing
                try:
                    params = self._parse_parameters(params_str.strip())
                    function_calls.append({
                        "function": func_name,
                        "parameters": params
                    })
                    logger.info(f"✅ Parsed XML function call with param parsing: {func_name} with params: {params}")
                except Exception as e:
                    logger.warning(f"Failed to parse XML parameters for {func_name}: {e}")
        
        if function_calls:
            logger.info(f"🎉 Successfully parsed {len(function_calls)} function calls from fallback patterns")
        else:
            logger.warning(f"❌ No function calls found in text: {text[:200]}...")
        
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
    
    def _extract_json_blocks(self, text: str) -> List[str]:
        """
        Extract potential JSON blocks from text.
        
        Handles cases where JSON might be:
        - Surrounded by other text
        - Within markdown code blocks
        - Mixed with regular text
        
        Args:
            text: Text to extract JSON from
            
        Returns:
            List of potential JSON strings
        """
        json_candidates = []
        
        # First, try the entire text if it looks like JSON
        stripped_text = text.strip()
        if stripped_text.startswith('{') and stripped_text.endswith('}'):
            json_candidates.append(stripped_text)
        
        # Extract JSON from markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        json_candidates.extend(code_blocks)
        
        # Extract standalone JSON objects from text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        json_candidates.extend(json_matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in json_candidates:
            candidate_clean = candidate.strip()
            if candidate_clean and candidate_clean not in seen:
                seen.add(candidate_clean)
                unique_candidates.append(candidate_clean)
        
        logger.debug(f"Extracted {len(unique_candidates)} JSON candidates from text")
        return unique_candidates
    
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
        original_text = text
        logger.debug(f"Cleaning text: {text[:100]}...")
        
        # First, extract and remove JSON blocks using the same method as parsing
        json_candidates = self._extract_json_blocks(text)
        
        for json_candidate in json_candidates:
            try:
                json_data = json.loads(json_candidate)
                
                # Check if this JSON contains function calls
                is_function_call_json = False
                
                # Format 1: { "function_call": { "name": "...", "parameters": {...} } }
                if "function_call" in json_data:
                    is_function_call_json = True
                
                # Format 2: { "FUNCTION_CALL": [ "..." ] }
                elif "FUNCTION_CALL" in json_data:
                    is_function_call_json = True
                
                # Format 3: {"function_name(params)"} - simple JSON with function call as key
                elif len(json_data) == 1:
                    for key in json_data.keys():
                        if re.match(r'\w+\(.*?\)', key.strip()):
                            is_function_call_json = True
                            break
                
                # If this is a function call JSON, remove it from text
                if is_function_call_json:
                    text = text.replace(json_candidate, "")
                    logger.debug(f"Removed function call JSON: {json_candidate}")
                    
            except json.JSONDecodeError:
                # If not valid JSON, check if it looks like incomplete JSON function call
                incomplete_pattern = r'^\{\s*"\w+\([^)]*\)"\s*\}$'
                if re.match(incomplete_pattern, json_candidate.strip()):
                    text = text.replace(json_candidate, "")
                    logger.debug(f"Removed incomplete function call JSON: {json_candidate}")
            except Exception as e:
                logger.debug(f"Error checking JSON candidate: {e}")
        
        # Remove markdown code blocks containing function calls
        code_block_pattern = r'```(?:json)?\s*\{[^}]*"function_call"[^}]*\}\s*```'
        text = re.sub(code_block_pattern, '', text, flags=re.DOTALL)
        
        # Remove other function call patterns
        
        # Remove FUNCTION_CALL patterns
        text = re.sub(r'FUNCTION_CALL:\s*\w+\([^)]*\)', '', text)
        
        # Remove [CALL] patterns
        text = re.sub(r'\[CALL\]\s*\w+\s*\{[^}]*\}', '', text)
        
        # Remove XML-like function call patterns
        text = re.sub(r'<function_call>.*?</function_call>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = re.sub(r'^\s+|\s+$', '', text)  # Strip leading/trailing whitespace
        text = text.strip()
        
        if text != original_text:
            logger.debug(f"Text cleaning complete. Original length: {len(original_text)}, cleaned length: {len(text)}")
        
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