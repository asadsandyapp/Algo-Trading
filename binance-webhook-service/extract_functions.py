#!/usr/bin/env python3
"""
Script to extract functions from binance_webhook_service.py into module files
"""
import re
import os

def find_function_end(lines, start_idx):
    """Find where a function ends by tracking indentation"""
    if start_idx >= len(lines):
        return len(lines)
    
    # Get the base indent of the function definition
    base_line = lines[start_idx]
    base_indent = len(base_line) - len(base_line.lstrip())
    
    # Skip the function definition line
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        
        # Empty lines and comments are part of the function
        if not line.strip() or line.strip().startswith('#'):
            continue
        
        # Check indentation
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line at the same or less indentation that's not a comment,
        # and it's not a continuation of the function (like elif, except, etc.)
        if current_indent <= base_indent:
            # Check if it's a control flow continuation
            stripped = line.strip()
            if stripped.startswith(('elif ', 'except ', 'else:', 'finally:')):
                continue
            # Check if it's a decorator
            if stripped.startswith('@'):
                continue
            # Otherwise, function has ended
            return i
    
    return len(lines)

def extract_function_code(lines, func_name, start_line):
    """Extract function code from lines"""
    start_idx = start_line - 1  # Convert to 0-based index
    
    # Find the function definition
    func_def_line = None
    for i in range(start_idx, min(start_idx + 5, len(lines))):
        if re.match(rf'^def {func_name}\(', lines[i]):
            func_def_line = i
            break
    
    if func_def_line is None:
        return None
    
    # Find where the function ends
    end_idx = find_function_end(lines, func_def_line)
    
    # Extract the function code
    func_code = '\n'.join(lines[func_def_line:end_idx])
    return func_code

# Read the main file
main_file = 'src/binance_webhook_service.py'
with open(main_file, 'r') as f:
    lines = f.readlines()
    # Remove newlines for processing, we'll add them back
    lines = [line.rstrip('\n') for line in lines]

# Find all function definitions
function_locations = {}
for i, line in enumerate(lines):
    match = re.match(r'^def (\w+)\(', line)
    if match:
        func_name = match.group(1)
        function_locations[func_name] = i + 1  # 1-based line number

# Functions to extract for AI validation
ai_functions = [
    'validate_entry2_standalone_with_ai',
    'parse_entry_analysis_from_reasoning', 
    'validate_signal_with_ai',
]

# Functions to extract for order management
order_functions = [
    'calculate_quantity',
    'delayed_tp_creation',
    'create_tp_if_needed',
    'create_single_tp_order',
    'create_tp1_tp2_if_needed',
    'update_trailing_stop_loss',
    'create_missing_tp_orders',
    'close_position_at_market',
    'create_limit_order',
]

print("Extracting functions...")

# Extract AI validation functions
ai_code_parts = []
for func_name in ai_functions:
    if func_name in function_locations:
        code = extract_function_code(lines, func_name, function_locations[func_name])
        if code:
            ai_code_parts.append(code)
            print(f"  ✓ Extracted {func_name} ({len(code.split(chr(10)))} lines)")
        else:
            print(f"  ✗ Failed to extract {func_name}")
    else:
        print(f"  ✗ Function {func_name} not found")

# Extract order management functions
order_code_parts = []
for func_name in order_functions:
    if func_name in function_locations:
        code = extract_function_code(lines, func_name, function_locations[func_name])
        if code:
            order_code_parts.append(code)
            print(f"  ✓ Extracted {func_name} ({len(code.split(chr(10)))} lines)")
        else:
            print(f"  ✗ Failed to extract {func_name}")
    else:
        print(f"  ✗ Function {func_name} not found")

print(f"\nExtracted {len(ai_code_parts)} AI functions")
print(f"Extracted {len(order_code_parts)} order functions")

# Write to files (we'll create the module files separately with proper imports)
print("\nFunction extraction complete. Module files will be created separately.")


