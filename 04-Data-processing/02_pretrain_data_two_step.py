"""
Batch processing of genome sequence files, supporting filtering, splitting, replacement, and padding operations, and saving the processed files to a new directory. These operations standardize the format of genome sequence data for downstream analysis and processing.
    Initialization: User sets processing directory, new storage directory, and other parameters in the main function.
    File traversal: Uses os.walk to traverse all files in the specified directory.
    File filtering: Checks if the filename meets the criteria via is_valid_filename.
    File processing: Calls process_file for files that meet the criteria.
    Output: Saves processed files to the new directory and prints processing and replacement information.
"""
import os
import random

# Function to generate a random ATGC sequence
def generate_random_atgc_sequence(length):
    bases = ['A', 'T', 'G', 'C']
    sequence = ''.join(random.choice(bases) for _ in range(length))
    return sequence

def is_valid_filename(filename, filter_num):
    # Check if filename matches the format and the number after the second "__" is greater than filter_num
    parts = filename.split("__")
    if len(parts) < 3:
        return False
    try:
        number = int(parts[2].split('.')[0])
        return number > filter_num
    except ValueError:
        return False

def replace_n_with_random_atgc(content):
    # Replace 'N' with a random character from 'ATGC'
    content_list = list(content)
    replacements = []
    for i, char in enumerate(content_list):
        if char == 'N':
            new_char = random.choice(['A', 'T', 'G', 'C'])
            content_list[i] = new_char
            replacements.append((i, new_char))
    return ''.join(content_list), replacements

def process_file(file_path, new_directory, original_directory, fill_short_sequences, sequence_length, filter_num, replace_n):
    # Read file content
    with open(file_path, 'r') as file:
        content = file.read()
    # Convert to uppercase
    # content = content.lower()
    content = content.upper()

    # Replace 'N' with random character (if replace_n is True)
    if replace_n:
        content, replacements = replace_n_with_random_atgc(content)
    else:
        replacements = []
    # Split every sequence_length characters
    lines = [content[i:i + sequence_length] for i in range(0, len(content), sequence_length)]
    # Check if the last line is shorter than sequence_length, and pad if fill_short_sequences is True
    for i in range(len(lines)):
        if len(lines[i]) < sequence_length and fill_short_sequences:
            lines[i] = lines[i] + generate_random_atgc_sequence(sequence_length - len(lines[i]))
    # Get the relative path of the original file to the original directory
    relative_path = os.path.relpath(file_path, start=original_directory)
    relative_dir, filename = os.path.split(relative_path)
    filename_without_ext, file_ext = os.path.splitext(filename)
    # Generate new suffix based on variables
    dynamic_suffix = f"__filter{filter_num}__fill{str(fill_short_sequences)}__len{sequence_length}__repN{replace_n}"
    new_filename = f"{filename_without_ext}{dynamic_suffix}{file_ext}"
    new_file_path = os.path.join(new_directory, relative_dir, new_filename)
    # Ensure the directory for the new file exists
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(new_file_path, 'w') as new_file:
        new_file.write('\n'.join(lines))
    print(f"Processed file: {new_file_path}")
    # Print replacement information
    if replace_n:
        print("Replacements:")
        for pos, new_char in replacements:
            print(f"Position {pos}: replaced 'N' with '{new_char}'")

def main(directory, new_directory, filter_num, fill_short_sequences, sequence_length, replace_n,random_seed):

    random.seed(random_seed)
    print(f"directory: {directory}, new_directory: {new_directory}, filter_num: {filter_num},sequence_length: {sequence_length},"
          f"fill_short_sequences: {fill_short_sequences}, replace_n: {replace_n}")
    print("#############################################################################")

    # Traverse all files in the specified directory
    for root, dirs, files in os.walk(directory):
        onestep_files_count = 0
        new_files_count = 0

        for file in files:
            onestep_files_count += 1
            # Check if filename matches the format
            if is_valid_filename(file, filter_num):
                file_path = os.path.join(root, file)
                process_file(file_path, new_directory, directory, fill_short_sequences, sequence_length, filter_num,
                             replace_n)
                new_files_count += 1  # Increment new file count for each successfully processed file

        # Print processing result for the current subdirectory
        relative_dir = os.path.relpath(root, start=directory)
        print(f"GENOME: {relative_dir}, onestep_files_count {onestep_files_count} files, Processed {new_files_count} new files.")
        print("#############################################################################")

if __name__ == "__main__":
    # User-defined parameters
    user_original_directory = "~/01_fasta_data/01_reads_output" # Directory to process
    user_new_directory = "~/01_fasta_data/02_reads_output"  # New storage directory
    user_filter_num = int(10000)  # Filter length
    user_sequence_length = 3060  # Sequence split length
    user_fill_short_sequences = False  # Whether to pad sequences shorter than the specified length
    user_replace_n = True  # Whether to replace 'N' with a random character
    random_seed_num = 12

    # Run the program
    main(user_original_directory, user_new_directory, user_filter_num, user_fill_short_sequences, user_sequence_length, user_replace_n, random_seed_num)

