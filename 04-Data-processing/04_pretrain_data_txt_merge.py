import os


def merge_txt_files(file_list, output_filename):
    """
    Merge the specified list of txt files into a single file.

    Args:
        file_list: List of txt file paths to merge
        output_filename: Output file name (with path) for the merged result
    """
    try:
        # If no txt files are provided, print a warning and exit
        if not file_list:
            print("Warning: No txt files provided for merging")
            return

        # Create the output file
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for filepath in file_list:
                try:
                    # Check if the file exists
                    if not os.path.exists(filepath):
                        print(f"Warning: File '{filepath}' does not exist, skipping")
                        continue

                    # Read each file and write to the output file
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        # Ensure file content ends with a newline (unless the file is completely empty)
                        if content and not content.endswith('\n'):
                            content += '\n'
                        outfile.write(content)

                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

        print(f"Successfully merged {len(file_list)} files into {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Set the list of files to merge

    input_files =[
        "~/01_fasta_data/03_reads_output/genome_ID1.txt",
        "~/01_fasta_data/03_reads_output/genome_ID2.txt",
                ]
    # Set the output file path
    output_file = "~/01_fasta_data/03_reads_output/genome-ID1-ID2.txt"

    # Call the merge function
    merge_txt_files(input_files, output_file)



