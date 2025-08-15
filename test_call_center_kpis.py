"""
Call Center KPI Test Suite
Comprehensive testing system for evaluating call center chatbot performance.
Tests function calling, response accuracy, and system performance metrics.
"""

import json
import time
import logging
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import statistics
import random
from dataclasses import dataclass, asdict
from functools import lru_cache
# Optional pandas import for Excel export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.translator.translator import Translator
from src.models.call_center_llm import CallCenterLLMManager
from src.models.call_center_llm_with_api import CallCenterLLMManagerWithAPI
from src.functions.function_caller import FunctionCaller
from src.functions.call_center_functions import FUNCTION_REGISTRY
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
@lru_cache(maxsize=1)
def get_translator():
    translator = Translator()
    return translator
@dataclass
class TestResult:
    """Data class for storing individual test results."""
    query: str
    expected_function: str
    detected_function: Optional[str]
    function_detected: bool
    parameters_correct: bool
    response_time: float
    response_text: str
    function_results: List[Dict[str, Any]]
    success: bool
    error: Optional[str]

@dataclass
class KPIMetrics:
    """Data class for storing KPI metrics."""
    total_tests: int
    function_detection_accuracy: float
    parameter_extraction_accuracy: float
    average_response_time: float
    median_response_time: float
    success_rate: float
    function_breakdown: Dict[str, Dict[str, Any]]
    performance_distribution: Dict[str, int]

class CallCenterKPITester:
    """
    Comprehensive KPI testing system for call center chatbot.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        test_questions_file: str = "sorular.json",
        customer_id_for_tests: str = "customer_001",
        use_api: bool = False
    ):
        """
        Initialize the KPI tester.
        
        Args:
            model_name: LLM model to test
            test_questions_file: JSON file containing test questions
            customer_id_for_tests: Default customer ID for testing
            use_api: Whether to use API version (CallCenterLLMManagerWithAPI) instead of local model
        """
        self.model_name = model_name
        self.test_questions_file = test_questions_file
        self.customer_id = customer_id_for_tests
        self.use_api = use_api
        self.results: List[TestResult] = []
        
        # Load test questions
        self.test_questions = self._load_test_questions()
        total_questions = sum(len(questions) for questions in self.test_questions.values())
        logger.info(f"Loaded {total_questions} test questions across {len(self.test_questions)} functions")
        
        # Print breakdown of questions per function
        for func_name, questions in self.test_questions.items():
            logger.info(f"  - {func_name}: {len(questions)} questions")
        
        # Initialize LLM manager (will be done when testing starts)
        self.llm_manager: Optional[CallCenterLLMManager] = None
        self.function_caller = FunctionCaller()
        
        logger.info(f"KPI Tester initialized - Using {'API' if use_api else 'Local'} LLM manager")
        
        # Test configuration
        self.test_config = {
            "max_length": 512,
            "temperature": 0.3,  # Lower temperature for more consistent results
            "top_p": 0.9
        }
        
        # Track request timing for rate limiting
        self._last_request_time = 0
    
    def _load_test_questions(self) -> Dict[str, List[str]]:
        """Load test questions from JSON file."""
        try:
            with open(self.test_questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # Clean up function names (remove typos)
            cleaned_questions = {}
            for func_name, question_list in questions.items():
                # Fix common typos in function names
                if func_name == "initaite_package_change":
                    func_name = "initiate_package_change"
                cleaned_questions[func_name] = question_list
            
            return cleaned_questions
        except Exception as e:
            logger.error(f"Error loading test questions: {e}")
            return {}
    
    def setup_llm(self, **kwargs) -> bool:
        """
        Setup the LLM manager for testing.
        
        Args:
            **kwargs: Additional arguments for LLM initialization
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            get_translator()
            if self.use_api:
                logger.info(f"Setting up API-based LLM: {self.model_name}")
                # Filter out kwargs that don't apply to API version
                api_kwargs = {k: v for k, v in kwargs.items() if k in ['api_key']}
                
                # Ensure we use a valid Google GenAI model
                if self.model_name == "microsoft/DialoGPT-medium" or "mistral" in self.model_name.lower():
                    self.model_name = "gemma-3-27b-it"
                    logger.info(f"Switched to valid Google GenAI model: {self.model_name}")
                
                self.llm_manager = CallCenterLLMManagerWithAPI(
                    model_name=self.model_name,
                    **api_kwargs
                )
                logger.info("API-based LLM setup completed successfully")
            else:
                logger.info(f"Setting up local LLM: {self.model_name}")
                self.llm_manager = CallCenterLLMManager(
                    model_name=self.model_name,
                    **kwargs
                )
                logger.info("Local LLM setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            return False
    
    def test_single_query(
        self,
        query: str,
        expected_function: str,
        add_customer_id: bool = True
    ) -> TestResult:
        """
        Test a single query and measure performance.
        
        Args:
            query: The query to test
            expected_function: Expected function to be called
            add_customer_id: Whether to add customer ID to response
            
        Returns:
            TestResult object containing test results
        """
        # Add customer ID context if needed (except for don't_know)
        if add_customer_id and expected_function not in ["don't_know"] and expected_function in [
            "get_customer_info", "check_billing_status", "process_payment", 
            "get_usage_summary", "create_support_ticket", "initiate_package_change"
        ]:
            # Add customer context to query
            query_with_context = f"Müşteri numarası: {self.customer_id}. {query}"
        else:
            query_with_context = query
        
        logger.debug(f"Testing query: {query_with_context}")
        
        error = None
        detected_function = None
        function_detected = False
        parameters_correct = False
        response_text = ""
        function_results = []
        response_time = 0.0
        
        # Retry logic for API errors and rate limiting
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Add delay between requests to avoid rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"    Retry attempt {attempt + 1}/{max_retries} after {delay:.1f}s delay...")
                    time.sleep(delay)
                elif hasattr(self, '_last_request_time'):
                    # Add small delay between requests to be respectful to API
                    time_since_last = time.time() - self._last_request_time
                    if time_since_last < 1.0:
                        time.sleep(1.0 - time_since_last)
                
                # Generate response (measure only the actual API call time)
                start_time = time.time()
                response_text, function_results = self.llm_manager.generate_response(
                    query_with_context,
                    **self.test_config
                )
                response_time = time.time() - start_time
                self._last_request_time = time.time()
                
                # Analyze function detection
                if function_results:
                    detected_function = function_results[0]["function"]
                    if expected_function == "don't_know":
                        # For don't_know, functions should NOT be detected
                        function_detected = False
                        parameters_correct = False
                    else:
                        function_detected = (detected_function == expected_function)
                        # Check parameter correctness
                        parameters_correct = self._validate_parameters(
                            detected_function,
                            function_results[0]["parameters"],
                            expected_function
                        )
                else:
                    # No functions detected
                    if expected_function == "don't_know":
                        # This is correct for don't_know questions
                        function_detected = True
                        parameters_correct = True
                    else:
                        function_detected = False
                        parameters_correct = False
                
                # If we get here, request was successful
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limiting or retryable error
                is_retryable = any(keyword in error_msg for keyword in [
                    'rate limit', 'quota', 'too many requests', 'timeout', 
                    'connection', 'service unavailable', '429', '503', '502', "I apologize, but I'm experiencing technical difficulties. Please try again or contact technical"
                ])
                
                if attempt < max_retries - 1 and is_retryable:
                    logger.warning(f"Retryable error on attempt {attempt + 1}: {e}")
                    continue
                else:
                    error = str(e)
                    logger.error(f"Error testing query '{query}' (attempt {attempt + 1}): {e}")
                    response_time = 0.0  # Set to 0 if all attempts failed
                    break
        
        # For don't_know, success means NO function was detected
        if expected_function == "don't_know":
            success = (not function_results and error is None)
        else:
            success = (function_detected and (error is None))
        
        return TestResult(
            query=query,
            expected_function=expected_function,
            detected_function=detected_function,
            function_detected=function_detected,
            parameters_correct=parameters_correct,
            response_time=response_time,
            response_text=response_text,
            function_results=function_results,
            success=success,
            error=error
        )
    
    def _validate_parameters(
        self,
        detected_function: str,
        parameters: Dict[str, Any],
        expected_function: str
    ) -> bool:
        """
        Validate if extracted parameters are reasonable.
        
        Args:
            detected_function: Function that was detected
            parameters: Parameters that were extracted
            expected_function: Expected function name
            
        Returns:
            bool: True if parameters seem correct
        """
        # For don't_know, no functions should be detected
        if expected_function == "don't_know":
            return False
        
        if detected_function != expected_function:
            return False
        
        # Get required parameters for the function
        if detected_function not in FUNCTION_REGISTRY:
            return False
        
        func_schema = FUNCTION_REGISTRY[detected_function]
        required_params = [
            param_name for param_name, param_info in func_schema["parameters"].items()
            if param_info.get("required", False)
        ]
        
        # Check if all required parameters are present
        for param in required_params:
            if param not in parameters:
                return False
        
        # Specific validation for common functions
        if detected_function == "get_customer_info":
            return "customer_id" in parameters
        
        elif detected_function == "process_payment":
            return (
                "customer_id" in parameters and
                "amount" in parameters and
                isinstance(parameters.get("amount"), (int, float)) and
                parameters["amount"] > 0
            )
        
        elif detected_function == "initiate_package_change":
            valid_packages = ["basic", "standard", "premium", "family"]
            return (
                "customer_id" in parameters and
                "new_package" in parameters and
                any(pkg in parameters["new_package"].lower() for pkg in valid_packages)
            )
        
        elif detected_function == "get_package_details":
            return "package_name" in parameters
        
        elif detected_function == "create_support_ticket":
            return (
                "customer_id" in parameters and
                "issue_type" in parameters and
                "description" in parameters
            )
        
        return True  # Default to True for other functions
    
    def run_comprehensive_test(self, sample_size: Optional[int] = None) -> KPIMetrics:
        """
        Run comprehensive testing on all test questions.
        
        Args:
            sample_size: Limit number of tests per function (None for all)
            
        Returns:
            KPIMetrics object containing comprehensive results
        """
        logger.info("Starting comprehensive KPI testing")
        
        if not self.llm_manager:
            raise RuntimeError("LLM not initialized. Call setup_llm() first.")
        
        self.results = []
        total_questions = 0
        
        # Print summary of what we're testing
        logger.info("=" * 50)
        logger.info("TEST SUMMARY:")
        for function_name, questions in self.test_questions.items():
            test_count = len(questions) if sample_size is None else min(sample_size, len(questions))
            logger.info(f"  {function_name}: {test_count}/{len(questions)} questions")
            total_questions += test_count
        logger.info(f"TOTAL: {total_questions} questions")
        logger.info("=" * 50)
        
        for function_name, questions in self.test_questions.items():
            if sample_size:
                questions = questions[:sample_size]
            
            logger.info(f"Testing function: {function_name} ({len(questions)} questions)")
            
            for i, question in enumerate(questions):
                logger.info(f"  Question {i+1}/{len(questions)}: {question}")
                
                result = self.test_single_query(question, function_name)
                result.response_text = get_translator().translate_en_to_tr(result.response_text)
                self.results.append(result)
                
                # Log result immediately
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                detected = result.detected_function or "None"
                
                if function_name == "don't_know":
                    # For don't_know, success means no function detected
                    expected_msg = "No function (correct)"
                    if not result.function_results:
                        logger.info(f"    → {status} | Expected: {expected_msg} | Detected: {detected}")
                    else:
                        logger.info(f"    → {status} | Expected: {expected_msg} | Detected: {detected} (should be None)")
                else:
                    logger.info(f"    → {status} | Expected: {function_name} | Detected: {detected}")
                
                # Progress logging
                if (i + 1) % 3 == 0:
                    logger.info(f"  Progress: {i+1}/{len(questions)} questions completed for {function_name}")
        
        logger.info(f"Completed testing {len(self.results)} questions")
        
        # Calculate and return KPIs
        return self._calculate_kpis()
    
    def _calculate_kpis(self) -> KPIMetrics:
        """Calculate comprehensive KPI metrics from test results."""
        if not self.results:
            raise ValueError("No test results available")
        
        total_tests = len(self.results)
        
        # Basic accuracy metrics
        function_detections = [r.function_detected for r in self.results]
        parameter_corrections = [r.parameters_correct for r in self.results if r.function_detected]
        response_times = [r.response_time for r in self.results]
        successes = [r.success for r in self.results]
        
        function_detection_accuracy = sum(function_detections) / total_tests * 100
        parameter_extraction_accuracy = (
            sum(parameter_corrections) / len(parameter_corrections) * 100 
            if parameter_corrections else 0
        )
        success_rate = sum(successes) / total_tests * 100
        
        # Response time metrics
        average_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        
        # Performance distribution (response time buckets)
        performance_distribution = {
            "< 1s": sum(1 for t in response_times if t < 1.0),
            "1-3s": sum(1 for t in response_times if 1.0 <= t < 3.0),
            "3-5s": sum(1 for t in response_times if 3.0 <= t < 5.0),
            "5-10s": sum(1 for t in response_times if 5.0 <= t < 10.0),
            "> 10s": sum(1 for t in response_times if t >= 10.0)
        }
        
        # Function-wise breakdown
        function_breakdown = {}
        for function_name in self.test_questions.keys():
            func_results = [r for r in self.results if r.expected_function == function_name]
            if func_results:
                func_detections = [r.function_detected for r in func_results]
                func_times = [r.response_time for r in func_results]
                func_successes = [r.success for r in func_results]
                
                # For don't_know, detection accuracy means correctly NOT detecting functions
                if function_name == "don't_know":
                    detection_accuracy = sum(1 for r in func_results if not r.function_results) / len(func_results) * 100
                else:
                    detection_accuracy = sum(func_detections) / len(func_detections) * 100
                
                function_breakdown[function_name] = {
                    "total_tests": len(func_results),
                    "detection_accuracy": detection_accuracy,
                    "avg_response_time": statistics.mean(func_times),
                    "success_rate": sum(func_successes) / len(func_successes) * 100
                }
        
        return KPIMetrics(
            total_tests=total_tests,
            function_detection_accuracy=function_detection_accuracy,
            parameter_extraction_accuracy=parameter_extraction_accuracy,
            average_response_time=average_response_time,
            median_response_time=median_response_time,
            success_rate=success_rate,
            function_breakdown=function_breakdown,
            performance_distribution=performance_distribution
        )
    
    def generate_detailed_report(self, kpis: KPIMetrics, save_path: str = "kpi_report.json") -> str:
        """
        Generate a detailed KPI report.
        
        Args:
            kpis: KPI metrics to include in report
            save_path: Path to save the report
            
        Returns:
            Formatted report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
========================================
CALL CENTER CHATBOT KPI REPORT
========================================
Generated: {timestamp}
Model: {self.model_name}
Total Tests: {kpis.total_tests}

OVERALL PERFORMANCE METRICS
========================================
✓ Function Detection Accuracy: {kpis.function_detection_accuracy:.2f}%
✓ Parameter Extraction Accuracy: {kpis.parameter_extraction_accuracy:.2f}%
✓ Overall Success Rate: {kpis.success_rate:.2f}%
✓ Average Response Time: {kpis.average_response_time:.2f}s
✓ Median Response Time: {kpis.median_response_time:.2f}s

RESPONSE TIME DISTRIBUTION
========================================
"""
        
        for time_bucket, count in kpis.performance_distribution.items():
            percentage = (count / kpis.total_tests) * 100
            report += f"  {time_bucket}: {count} tests ({percentage:.1f}%)\n"
        
        report += "\nFUNCTION-WISE BREAKDOWN\n"
        report += "========================================\n"
        
        for func_name, metrics in kpis.function_breakdown.items():
            report += f"\n{func_name.upper()}:\n"
            report += f"  • Tests: {metrics['total_tests']}\n"
            report += f"  • Detection: {metrics['detection_accuracy']:.2f}%\n"
            report += f"  • Avg Time: {metrics['avg_response_time']:.2f}s\n"
            report += f"  • Success: {metrics['success_rate']:.2f}%\n"
        
        # Add failure analysis
        failures = [r for r in self.results if not r.success]
        if failures:
            report += f"\nFAILURE ANALYSIS ({len(failures)} failures)\n"
            report += "========================================\n"
            
            failure_reasons = {}
            for failure in failures:
                if failure.error:
                    reason = "System Error"
                elif not failure.function_detected:
                    reason = "Function Not Detected"
                elif not failure.parameters_correct:
                    reason = "Parameter Extraction Failed"
                else:
                    reason = "Unknown"
                
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in failure_reasons.items():
                percentage = (count / len(failures)) * 100
                report += f"  • {reason}: {count} ({percentage:.1f}%)\n"
        
        # Save detailed report
        detailed_data = {
            "timestamp": timestamp,
            "model": self.model_name,
            "kpis": asdict(kpis),
            "test_results": [asdict(r) for r in self.results]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        report += f"\n\nDetailed results saved to: {save_path}\n"
        return report
    
    def export_to_excel(self, filename: str = "kpi_results.xlsx"):
        """
        Export test results to Excel for detailed analysis.
        
        Args:
            filename: Excel filename to save
        """
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available. Skipping Excel export. Install pandas with: pip install pandas openpyxl")
            return
        
        if not self.results:
            logger.warning("No results to export")
            return
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                "Query": result.query,
                "Expected Function": result.expected_function,
                "Detected Function": result.detected_function,
                "Function Detected": result.function_detected,
                "Parameters Correct": result.parameters_correct,
                "Response Time (s)": result.response_time,
                "Success": result.success,
                "Error": result.error or "",
                "Response Text": result.response_text[:100] + "..." if len(result.response_text) > 100 else result.response_text
            })
        
        df = pd.DataFrame(data)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results
            df.to_excel(writer, sheet_name='Test Results', index=False)
            
            # Summary by function
            summary_data = []
            for func_name in self.test_questions.keys():
                func_results = df[df['Expected Function'] == func_name]
                if not func_results.empty:
                    summary_data.append({
                        'Function': func_name,
                        'Total Tests': len(func_results),
                        'Detection Accuracy (%)': func_results['Function Detected'].mean() * 100,
                        'Parameter Accuracy (%)': func_results['Parameters Correct'].mean() * 100,
                        'Success Rate (%)': func_results['Success'].mean() * 100,
                        'Avg Response Time (s)': func_results['Response Time (s)'].mean(),
                        'Max Response Time (s)': func_results['Response Time (s)'].max()
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Function Summary', index=False)
        
        logger.info(f"Results exported to {filename}")

def main():
    """Main function to run KPI tests for API version only."""
    logger.info("Starting Call Center KPI Testing - API Version")
    
    try:
        # Initialize tester for API version
        tester = CallCenterKPITester(
            model_name="gemma-3-27b-it",  # API model
            test_questions_file="sorular.json",
            use_api=True
        )
        
        # Setup API LLM
        success = tester.setup_llm()
        
        if not success:
            logger.error("Failed to setup API LLM. Exiting.")
            return
        
        # Run comprehensive tests
        logger.info("Running comprehensive KPI tests for API LLM...")
        kpis = tester.run_comprehensive_test(sample_size=None)  # Test ALL questions from sorular.json
        
        # Generate and display report
        report = tester.generate_detailed_report(kpis, save_path="kpi_report_api.json")
        print(report)
        
        # Export to Excel
        tester.export_to_excel("call_center_kpi_results_api.xlsx")
        
        logger.info("KPI testing for API LLM completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    main()
