"""
This script converts JSON files to CSV format with two different conversion methods:
1. Basic method: Converts JSON objects to CSV with headers as keys and rows as values
2. Advanced method: Saves keys in first row and corresponding values in second row
"""

import json
import csv
import os
from pathlib import Path


def convert_json_to_csv(file_path):
    """
    Read a JSON file and convert it to CSV format
    """
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()

    # Parse content, handle possible format variations
    # If content is stored directly as dictionary format, special handling is needed
    if content.startswith("{") and content.endswith("}"):
        # Direct dictionary format
        data = eval(content)  # Note: In actual applications, json.loads() should be used, but here we handle possible non-standard formats
        try:
            data = json.loads(content)
        except:
            # If json.loads fails, try using eval (only use on trusted data)
            data = eval(content)
    else:
        # If it's multi-line format with one dictionary per line
        lines = content.strip().split('\n')
        data_list = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except:
                    data = eval(line)
                    data_list.append(data)

    # Determine output file path
    dir_path = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(dir_path, f"{base_name}_converted.csv")

    # Handle single dictionary case
    if isinstance(data, dict):
        data_list = [data]

    # Write to CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header (key names)
        writer.writeheader()

        # Write data rows (values)
        for data in data_list:
            writer.writerow(data)

    print(f"Conversion completed! CSV file saved to: {output_file_path}")
    return output_file_path


def convert_json_to_csv_advanced(file_path):
    """
    Read a JSON file and convert it to CSV format, saved in specified format
    First row contains key names, second row contains corresponding values
    """
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()

    # Parse content
    try:
        # Try to parse with json.loads
        data = json.loads(content)
    except json.JSONDecodeError:
        # If it fails, try parsing with eval (only for trusted data)
        data = eval(content)

    # Determine output file path
    dir_path = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(dir_path, f"{base_name}_2_csv.csv")

    # Prepare CSV data
    keys = list(data.keys())
    values = list(data.values())

    # Write to CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write first row: key names
        writer.writerow(keys)

        # Write second row: corresponding values
        writer.writerow(values)

    print(f"Conversion completed! CSV file saved to: {output_file_path}")
    print(f"Key names: {keys}")
    print(f"Corresponding values: {values}")
    return output_file_path


def process_all_test_metrics_json(directory_path, conversion_type="advanced"):
    """
    Traverse all test_metrics.json files in subdirectories of the specified directory and process them

    Args:
        directory_path (str): Root directory path to search
        conversion_type (str): Conversion type, "basic" or "advanced"
    """
    # Convert path to Path object
    root_path = Path(directory_path)

    if not root_path.exists():
        print(f"Error: Directory {directory_path} does not exist!")
        return

    # Find all files named test_metrics.json
    json_files = list(root_path.rglob("test_metrics.json"))

    if not json_files:
        print(f"No test_metrics.json files found in directory {directory_path} and its subdirectories")
        return

    print(f"Found {len(json_files)} test_metrics.json files:")
    for i, file_path in enumerate(json_files, 1):
        print(f"{i}. {file_path}")

    # Process each found file
    success_count = 0
    error_count = 0

    for file_path in json_files:
        print(f"\nProcessing: {file_path}")
        try:
            if conversion_type == "basic":
                convert_json_to_csv(str(file_path))
            else:  # Default to advanced
                convert_json_to_csv_advanced(str(file_path))
            success_count += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            error_count += 1

    print(f"\nProcessing completed!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {error_count} files")


def process_test_metrics_in_directory(directory_path):
    """
    Function specifically for processing test_metrics.json files, using advanced conversion method
    """
    process_all_test_metrics_json(directory_path, conversion_type="advanced")


def process_test_metrics_in_directory_basic(directory_path):
    """
    Function specifically for processing test_metrics.json files, using basic conversion method
    """
    process_all_test_metrics_json(directory_path, conversion_type="basic")


# If running this script directly
if __name__ == "__main__":
    # Example usage
    # You can specify directory path directly, or use interactive input
    directory_path = r"./01_sort_out_result_files"

    if os.path.exists(directory_path):
        # Use advanced conversion method to process all test_metrics.json files
        process_test_metrics_in_directory(directory_path)
    else:
        print("Directory does not exist, please check if the path is correct!")

    # If you need to process a single file, uncomment the following lines
    # file_path = "test_metrics.json"
    # if os.path.exists(file_path):
    #     convert_json_to_csv_advanced(file_path)
    # else:
    #     print("File does not exist, please check if the path is correct!")