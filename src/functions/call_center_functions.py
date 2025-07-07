"""
Call Center Functions
Mock functions for a call center chatbot that can handle customer service operations.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

# Mock customer database
MOCK_CUSTOMERS = {
    "customer_001": {
        "name": "John Doe",
        "phone": "+1-555-0123",
        "email": "john.doe@email.com",
        "account_status": "active",
        "current_package": "premium",
        "billing_address": "123 Main St, Anytown, ST 12345",
        "join_date": "2023-01-15",
        "last_payment": "2025-07-01",
        "next_billing_date": "2025-08-01",
        "outstanding_balance": 0.0
    },
    "customer_002": {
        "name": "Jane Smith", 
        "phone": "+1-555-0456",
        "email": "jane.smith@email.com",
        "account_status": "active",
        "current_package": "basic",
        "billing_address": "456 Oak Ave, Somewhere, ST 67890",
        "join_date": "2022-06-20",
        "last_payment": "2025-06-11",
        "next_billing_date": "2025-07-30",
        "outstanding_balance": 25.99
    },
    "customer_003": {
        "name": "Bob Johnson",
        "phone": "+1-555-0789",
        "email": "bob.johnson@email.com", 
        "account_status": "suspended",
        "current_package": "standard",
        "billing_address": "789 Pine St, Elsewhere, ST 11111",
        "join_date": "2023-03-10",
        "last_payment": "2025-04-30",
        "next_billing_date": "2025-07-30",
        "outstanding_balance": 75.99
    }
}

# Mock package definitions
AVAILABLE_PACKAGES = {
    "basic": {
        "name": "Basic Plan",
        "price": 19.99,
        "features": ["10GB Data", "Unlimited Calls", "SMS", "Basic Support"],
        "description": "Perfect for light users who need essential connectivity"
    },
    "standard": {
        "name": "Standard Plan", 
        "price": 39.99,
        "features": ["50GB Data", "Unlimited Calls", "SMS", "Priority Support", "Mobile Hotspot"],
        "description": "Great for regular users who need more data and features"
    },
    "premium": {
        "name": "Premium Plan",
        "price": 59.99,
        "features": ["Unlimited Data", "Unlimited Calls", "SMS", "24/7 Premium Support", "Mobile Hotspot", "International Roaming", "Streaming Benefits"],
        "description": "Ultimate plan for heavy users with all premium features"
    },
    "family": {
        "name": "Family Plan",
        "price": 99.99,
        "features": ["Unlimited Data (4 lines)", "Unlimited Calls", "SMS", "Family Controls", "Shared Data Pool", "Multi-device Support"],
        "description": "Perfect for families with multiple users and devices"
    }
}

def get_customer_info(customer_id: str) -> Dict[str, Any]:
    """
    Retrieve customer information by customer ID.
    
    Args:
        customer_id: The customer's unique identifier
        
    Returns:
        Dictionary containing customer information
    """
    logger.info(f"Looking up customer info for: {customer_id}")
    
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    customer = MOCK_CUSTOMERS[customer_id].copy()
    return {
        "success": True,
        "error": None,
        "data": customer
    }

def get_available_packages() -> Dict[str, Any]:
    """
    Get all available service packages.
    
    Returns:
        Dictionary containing all available packages
    """
    logger.info("Retrieving available packages")
    
    return {
        "success": True,
        "error": None,
        "data": AVAILABLE_PACKAGES
    }

def get_package_details(package_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific package.
    
    Args:
        package_name: Name of the package to get details for
        
    Returns:
        Dictionary containing package details
    """
    logger.info(f"Getting package details for: {package_name}")
    
    if package_name not in AVAILABLE_PACKAGES:
        return {
            "success": False,
            "error": f"Package '{package_name}' not found",
            "data": None
        }
    
    package = AVAILABLE_PACKAGES[package_name].copy()
    return {
        "success": True,
        "error": None,
        "data": package
    }

def initiate_package_change(customer_id: str, new_package: str, effective_date: str = "") -> Dict[str, Any]:
    """
    Initiate a package change for a customer.
    
    Args:
        customer_id: The customer's unique identifier
        new_package: The new package to switch to
        effective_date: When the change should take effect (optional, defaults to next billing cycle)
        
    Returns:
        Dictionary containing the change request details
    """
    new_package = new_package.lower()
    new_package = new_package.replace("plan", "")
    new_package = new_package.strip()
    logger.info(f"Initiating package change for customer {customer_id} to {new_package}")
    
    # Check if customer exists
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    # Check if package exists
    if new_package not in AVAILABLE_PACKAGES:
        return {
            "success": False,
            "error": f"Package '{new_package}' not found",
            "data": None
        }
    
    customer = MOCK_CUSTOMERS[customer_id]
    
    # Check if customer is trying to switch to same package
    if customer["current_package"] == new_package:
        return {
            "success": False,
            "error": f"Customer is already on the {new_package} package",
            "data": None
        }
    
    # Generate change request
    if effective_date is "":
        effective_date = customer["next_billing_date"]
    
    change_request = {
        "request_id": f"CHG-{random.randint(100000, 999999)}",
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "old_package": customer["current_package"],
        "new_package": new_package,
        "effective_date": effective_date,
        "price_change": AVAILABLE_PACKAGES[new_package]["price"] - AVAILABLE_PACKAGES[customer["current_package"]]["price"],
        "status": "pending",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return {
        "success": True,
        "error": None,
        "data": change_request
    }

def check_billing_status(customer_id: str) -> Dict[str, Any]:
    """
    Check the billing status for a customer.
    
    Args:
        customer_id: The customer's unique identifier
        
    Returns:
        Dictionary containing billing status information
    """
    logger.info(f"Checking billing status for customer: {customer_id}")
    
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    customer = MOCK_CUSTOMERS[customer_id]
    current_package = AVAILABLE_PACKAGES[customer["current_package"]]
    
    billing_info = {
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "current_package": customer["current_package"],
        "monthly_charge": current_package["price"],
        "last_payment_date": customer["last_payment"],
        "next_billing_date": customer["next_billing_date"],
        "outstanding_balance": customer["outstanding_balance"],
        "account_status": customer["account_status"],
        "payment_status": "current" if customer["outstanding_balance"] == 0 else "past_due"
    }
    
    return {
        "success": True,
        "error": None,
        "data": billing_info
    }

def process_payment(customer_id: str, amount: float, payment_method: str = "credit_card") -> Dict[str, Any]:
    """
    Process a payment for a customer.
    
    Args:
        customer_id: The customer's unique identifier
        amount: Payment amount
        payment_method: Payment method (credit_card, debit_card, bank_transfer)
        
    Returns:
        Dictionary containing payment processing results
    """
    logger.info(f"Processing payment of ${amount} for customer: {customer_id}")
    
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    if amount <= 0:
        return {
            "success": False,
            "error": "Payment amount must be greater than 0",
            "data": None
        }
    
    # Simulate payment processing
    payment_result = {
        "transaction_id": f"TXN-{random.randint(1000000, 9999999)}",
        "customer_id": customer_id,
        "amount": amount,
        "payment_method": payment_method,
        "status": "successful",
        "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confirmation_number": f"CONF-{random.randint(100000, 999999)}"
    }
    
    # Update customer balance (in a real system, this would update the database)
    customer = MOCK_CUSTOMERS[customer_id]
    new_balance = max(0, customer["outstanding_balance"] - amount)
    
    payment_result["previous_balance"] = customer["outstanding_balance"]
    payment_result["new_balance"] = new_balance
    
    return {
        "success": True,
        "error": None,
        "data": payment_result
    }

def get_usage_summary(customer_id: str, period: str = "current_month") -> Dict[str, Any]:
    """
    Get usage summary for a customer.
    
    Args:
        customer_id: The customer's unique identifier
        period: Period to get usage for (current_month, last_month, last_3_months)
        
    Returns:
        Dictionary containing usage information
    """
    logger.info(f"Getting usage summary for customer {customer_id}, period: {period}")
    
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    customer = MOCK_CUSTOMERS[customer_id]
    
    # Generate mock usage data
    usage_data = {
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "package": customer["current_package"],
        "period": period,
        "data_usage": {
            "used_gb": round(random.uniform(5, 45), 2),
            "included_gb": "Unlimited" if customer["current_package"] == "premium" else random.choice([10, 50, 100]),
            "overage_charges": 0.0
        },
        "call_usage": {
            "minutes_used": random.randint(200, 1500),
            "included_minutes": "Unlimited",
            "overage_charges": 0.0
        },
        "sms_usage": {
            "messages_sent": random.randint(50, 500),
            "included_messages": "Unlimited",
            "overage_charges": 0.0
        }
    }
    
    return {
        "success": True,
        "error": None,
        "data": usage_data
    }

def create_support_ticket(customer_id: str, issue_type: str, description: str, priority: str = "medium") -> Dict[str, Any]:
    """
    Create a support ticket for a customer.
    
    Args:
        customer_id: The customer's unique identifier
        issue_type: Type of issue (technical, billing, account, general)
        description: Description of the issue
        priority: Priority level (low, medium, high, urgent)
        
    Returns:
        Dictionary containing ticket information
    """
    logger.info(f"Creating support ticket for customer {customer_id}")
    
    if customer_id not in MOCK_CUSTOMERS:
        return {
            "success": False,
            "error": f"Customer {customer_id} not found",
            "data": None
        }
    
    customer = MOCK_CUSTOMERS[customer_id]
    
    ticket = {
        "ticket_id": f"TKT-{random.randint(100000, 999999)}",
        "customer_id": customer_id,
        "customer_name": customer["name"],
        "issue_type": issue_type,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assigned_agent": f"Agent {random.randint(1, 10)}",
        "estimated_resolution": (datetime.now() + timedelta(days=random.randint(1, 3))).strftime("%Y-%m-%d")
    }
    
    return {
        "success": True,
        "error": None,
        "data": ticket
    }

# Function registry for easy lookup
FUNCTION_REGISTRY = {
    "get_customer_info": {
        "function": get_customer_info,
        "description": "Get customer information by customer ID",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True}
        }
    },
    "get_available_packages": {
        "function": get_available_packages,
        "description": "Get all available service packages",
        "parameters": {}
    },
    "get_package_details": {
        "function": get_package_details,
        "description": "Get detailed information about a specific package",
        "parameters": {
            "package_name": {"type": "string", "description": "Name of the package", "required": True}
        }
    },
    "initiate_package_change": {
        "function": initiate_package_change,
        "description": "Initiate a package change for a customer",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True},
            "new_package": {"type": "string", "description": "New package to switch to", "required": True},
            "effective_date": {"type": "string", "description": "When change should take effect (YYYY-MM-DD)", "required": False}
        }
    },
    "check_billing_status": {
        "function": check_billing_status,
        "description": "Check billing status for a customer",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True}
        }
    },
    "process_payment": {
        "function": process_payment,
        "description": "Process a payment for a customer",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True},
            "amount": {"type": "number", "description": "Payment amount", "required": True},
            "payment_method": {"type": "string", "description": "Payment method", "required": False}
        }
    },
    "get_usage_summary": {
        "function": get_usage_summary,
        "description": "Get usage summary for a customer",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True},
            "period": {"type": "string", "description": "Period to get usage for", "required": False}
        }
    },
    "create_support_ticket": {
        "function": create_support_ticket,
        "description": "Create a support ticket for a customer",
        "parameters": {
            "customer_id": {"type": "string", "description": "Customer's unique identifier", "required": True},
            "issue_type": {"type": "string", "description": "Type of issue", "required": True},
            "description": {"type": "string", "description": "Description of the issue", "required": True},
            "priority": {"type": "string", "description": "Priority level", "required": False}
        }
    }
} 