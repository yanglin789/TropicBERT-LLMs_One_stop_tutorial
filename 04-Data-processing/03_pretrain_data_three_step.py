"""
Script description: This script is used to merge all text files (.txt) in eligible subdirectories under the specified main directory,
and save the merged result to the specified output directory. For each eligible subdirectory, the text files are merged in order
(sorted by a specific number embedded in the filename), and the merged file is named "subdirectoryname_merged.txt".
New feature: The merged file can be read, lines can be shuffled (optional), and all content can be output to a new file.
"""
import os
import re
import random


def extract_sort_key(file_name):
    """Extract sort key from filename, try multiple patterns to ensure all files are handled"""

    # Try to extract pattern like NC_045127.1__55558335 (first number sequence)
    match = re.search(r'__(NC_\d+\.\d+)__(\d+)', file_name)
    if match:
        return (match.group(1), int(match.group(2)))

    # Try to extract pattern like NW_022204028.1__63246
    match = re.search(r'__(NW_\d+\.\d+)__(\d+)', file_name)
    if match:
        return (match.group(1), int(match.group(2)))

    # Try to extract pattern like LG05__36727471
    match = re.search(r'__(LG\d+)__(\d+)', file_name)
    if match:
        return (match.group(1), int(match.group(2)))

    # Try to extract pattern like ctg000030__100239
    match = re.search(r'__(ctg\d+)__(\d+)', file_name)
    if match:
        return (match.group(1), int(match.group(2)))

    # If none of the above patterns match, try to extract the last number sequence in the filename as the sort key
    numbers = re.findall(r'\d+', file_name)
    if numbers:
        return int(numbers[-1])

    # Default: return the filename itself to ensure no file is filtered out
    return file_name


def merge_files_in_directory(directory, output_filepath):
    # Get all txt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Sort files by extracted sort key (no files are filtered)
    sorted_files = sorted(files, key=lambda f: extract_sort_key(f))

    # Merge files
    with open(output_filepath, 'w') as outfile:
        for filename in sorted_files:
            filepath = os.path.join(directory, filename)
            print(f"Merging file: {filename}")  # Print the filename being merged
            with open(filepath, 'r') as infile:
                outfile.write(infile.read())
                outfile.write('\n')  # Add newline to separate file contents


def shuffle_file_lines(input_filepath, output_filepath):
    """Read file, shuffle lines, write to new file"""
    with open(input_filepath, 'r') as f:
        lines = f.readlines()

    # Shuffle lines
    random.shuffle(lines)

    with open(output_filepath, 'w') as f:
        f.writelines(lines)


def process_merged_files(output_directory, final_output_filepath, shuffle=False):
    """Process all merged files, optionally shuffle lines"""
    # Get all merged files
    merged_files = [f for f in os.listdir(output_directory) if f.endswith('_merged.txt')]

    with open(final_output_filepath, 'w') as outfile:
        for merged_file in merged_files:
            input_path = os.path.join(output_directory, merged_file)
            print(f"Processing merged file: {merged_file}")

            if shuffle:
                # Temporary file path
                temp_path = os.path.join(output_directory, f"temp_{merged_file}")
                shuffle_file_lines(input_path, temp_path)
                with open(temp_path, 'r') as f:
                    outfile.write(f.read())
                # Delete temporary file
                os.remove(temp_path)
            else:
                with open(input_path, 'r') as f:
                    outfile.write(f.read())

            # Add separator between different files
            outfile.write('\n' + '=' * 80 + '\n\n')


def process_main_directory(main_directory, output_directory, prefix, suffix,
                           final_output_filename="all_merged_final.txt", shuffle=False):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Traverse all subdirectories under the main directory
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith(prefix) and subdir.endswith(suffix):
            print(f"####################################################################################")
            print(f"Processing directory: {subdir}")
            # Set output file path, save in the specified output directory
            output_filepath = os.path.join(output_directory, f'{subdir}_merged.txt')
            merge_files_in_directory(subdir_path, output_filepath)

    # Process all merged files
    final_output_path = os.path.join(output_directory, final_output_filename)
    process_merged_files(output_directory, final_output_path, shuffle)
    print(f"All merged files have been processed and saved to: {final_output_path}")


if __name__ == "__main__":
    # User-defined variables
    main_dir = "~/01_fasta_data/02_reads_output"  # Main directory path
    out_dir = "~/01_fasta_data/03_reads_output"  # Output directory path

    prefix_str = "TP" # genome directory name prefix
    suffix_str = "reads"
    final_output_name = "all_species_merged_final_UNshuffle.txt"  # Name of the final merged file
    shuffle_lines = False  # Whether to shuffle lines

    # Execute merge operation
    process_main_directory(main_dir, out_dir, prefix_str, suffix_str,
                           final_output_name, shuffle_lines)