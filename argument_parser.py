"""
Command-line argument parser for MSCI inclusion and digital transformation analysis
"""

import argparse
import os
from pathlib import Path

def parse_arguments():
    """
    Parse command-line arguments for the analysis

    Returns:
    --------
    argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MSCI inclusion and digital transformation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add subparsers for different operation modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Preprocess mode
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data only')
    preprocess_parser.add_argument('--data', type=str,
                        help='Path to raw data file (.dta or .csv)')
    preprocess_parser.add_argument('--output', type=str, default='dataset/preprocessed_data.pkl',
                        help='Path to save preprocessed data')
    preprocess_parser.add_argument('--format', type=str, choices=['dta', 'csv'], default='dta',
                        hehlp='Input file format (dta for Stata, csv for CSV)')
    
    # Analyze mode
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on preprocessed data')
    analyze_parser.add_argument('--preprocessed', type=str, default='dataset/preprocessed_data.pkl',
                        help='Path to preprocessed data file')
    analyze_parser.add_argument('--steps', type=str, nargs='+', 
                        choices=['descriptive', 'hypothesis', 'models', 'mechanisms', 'all'],
                        default=['all'],
                        help='Analysis steps to run')
    
    # Full mode (preprocess + analyze)
    full_parser = subparsers.add_parser('full', help='Preprocess data and run full analysis')
    full_parser.add_argument('--data', type=str,
                        help='Path to raw data file (.dta or .csv)')
    full_parser.add_argument('--format', type=str, choices=['dta', 'csv'], default='dta',
                        help='Input file format (dta for Stata, csv for CSV)')
    full_parser.add_argument('--steps', type=str, nargs='+', 
                        choices=['descriptive', 'hypothesis', 'models', 'mechanisms', 'all'],
                        default=['all'],
                        help='Analysis steps to run')
    
    # Set default mode to "full" for backward compatibility
    parser.set_defaults(mode='full')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if args is None:
        parser.print_help()
        return None
        
    return args

def validate_file_path(file_path, expected_ext=None):
    """
    Validate that the file exists and has the expected extension
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    expected_ext : str or None
        Expected file extension (without dot)
        
    Returns:
    --------
    bool: True if the file is valid, False otherwise
    """
    if file_path is None:
        return False
        
    path = Path(file_path)
    
    # Check if the file exists
    if not path.exists():
        print(f"Error: File {path} does not exist")
        return False
        
    # Check the extension if specified
    if expected_ext and path.suffix.lower() != f".{expected_ext.lower()}":
        print(f"Error: Expected file with .{expected_ext} extension, got {path}")
        return False
        
    return True
