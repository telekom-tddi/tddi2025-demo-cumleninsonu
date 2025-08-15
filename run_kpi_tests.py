#!/usr/bin/env python3
"""
Quick script to run KPI tests for the call center system.
This script provides an easy way to test the system with different configurations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from test_call_center_kpis import CallCenterKPITester

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    for handler in logging.root.handlers[:]:
        print(handler)
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'kpi_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Run KPI tests for call center system')
    
    parser.add_argument(
        '--model', 
        default='mistralai/Mistral-7B-Instruct-v0.3',
        help='Model name to test (default: mistralai/Mistral-7B-Instruct-v0.3 for local, gemma-3-27b-it for API)'
    )
    
    parser.add_argument(
        '--use-api',
        action='store_true',
        default=True,
        help='Use API-based LLM manager (CallCenterLLMManagerWithAPI) instead of local model (default: True)'
    )
    
    parser.add_argument(
        '--use-local',
        action='store_true',
        help='Use local LLM manager instead of API (overrides --use-api)'
    )
    
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run model on (default: cpu)'
    )
    
    parser.add_argument(
        '--quantization',
        type=int,
        choices=[4, 8],
        help='Quantization bits (4 or 8) for memory efficiency'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of questions to test per function (default: None = all questions)'
    )
    
    parser.add_argument(
        '--customer-id',
        default='customer_001',
        help='Customer ID to use for testing (default: customer_001)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='test_results',
        help='Directory to save test results (default: test_results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Override use_api if use_local is specified
    if args.use_local:
        args.use_api = False
    
    # Adjust model name based on LLM type
    if args.use_api and args.model == 'microsoft/DialoGPT-medium':
        args.model = 'gemma-3-27b-it'  # Default API model
    
    logger.info("=" * 60)
    logger.info("CALL CENTER KPI TESTING")
    logger.info("=" * 60)
    logger.info(f"LLM Type: {'API' if args.use_api else 'Local'}")
    logger.info(f"Model: {args.model}")
    if not args.use_api:
        logger.info(f"Device: {args.device}")
        logger.info(f"Quantization: {args.quantization or 'None'}")
    logger.info(f"Sample size: {args.sample_size} questions per function")
    logger.info(f"Customer ID: {args.customer_id}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    try:
        # Initialize tester
        logger.info("Initializing KPI tester...")
        tester = CallCenterKPITester(
            model_name=args.model,
            test_questions_file="sorular.json",
            customer_id_for_tests=args.customer_id,
            use_api=args.use_api
        )
        
        # Setup LLM
        logger.info("Setting up LLM...")
        if args.use_api:
            setup_kwargs = {}  # API doesn't need device/quantization params
        else:
            setup_kwargs = {
                'device': args.device
            }
            if args.quantization:
                setup_kwargs['load_in_bits'] = args.quantization
        
        success = tester.setup_llm(**setup_kwargs)
        
        if not success:
            logger.error("Failed to setup LLM. Exiting.")
            return 1
        
        logger.info("LLM setup completed successfully!")
        
        # Run tests
        logger.info("Starting KPI tests...")
        start_time = datetime.now()
        
        kpis = tester.run_comprehensive_test(sample_size=args.sample_size)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"KPI testing completed in {total_time:.2f} seconds!")
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_type = "api" if args.use_api else "local"
        
        # Text report
        report_path = os.path.join(args.output_dir, f"kpi_report_{llm_type}_{timestamp}.json")
        report = tester.generate_detailed_report(kpis, save_path=report_path)
        
        # Excel export
        excel_path = os.path.join(args.output_dir, f"kpi_results_{llm_type}_{timestamp}.xlsx")
        tester.export_to_excel(excel_path)
        
        # Save text report
        text_report_path = os.path.join(args.output_dir, f"kpi_summary_{llm_type}_{timestamp}.txt")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Display summary
        print("\n" + "=" * 60)
        print("KPI TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✓ Total Tests: {kpis.total_tests}")
        print(f"✓ Function Detection Accuracy: {kpis.function_detection_accuracy:.2f}%")
        print(f"✓ Parameter Extraction Accuracy: {kpis.parameter_extraction_accuracy:.2f}%")
        print(f"✓ Overall Success Rate: {kpis.success_rate:.2f}%")
        print(f"✓ Average Response Time: {kpis.average_response_time:.2f}s")
        print(f"✓ Test Duration: {total_time:.2f}s")
        print("\nFiles Generated:")
        print(f"  • JSON Report: {report_path}")
        print(f"  • Excel Results: {excel_path}")
        print(f"  • Text Summary: {text_report_path}")
        print("=" * 60)
        
        # Print detailed report to console
        print(report)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
