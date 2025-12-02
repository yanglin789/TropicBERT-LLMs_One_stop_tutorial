import os
import re


def process_genome_file(f1_file_path, f1_output_path):
    """Process genome FASTA file, extract each read to a separate file
    
    Args:
        f1_file_path (str): Input FASTA file path
        f1_output_path (str): Output directory path
    """
    # 1. Check if the file is a FASTA file
    if not f1_file_path.endswith(".fasta"):
        print("File does not have the required .fasta suffix.")
        return

    # Extract genome_name (assume format {genome_name}_*.fasta)
    genome_name = os.path.basename(f1_file_path).split('_', 1)[0]
    print(f"##############################################################################")
    print(f"genome_name: {genome_name}")

    # Create output folder
    folder_name = os.path.join(f1_output_path, f"{genome_name}_reads")
    os.makedirs(folder_name, exist_ok=True)
    print(f"folder_name: {folder_name}")

    try:
        with open(f1_file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
            
        reads_num = 0
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if it is a header line (starts with '>')
            if line.startswith('>'):
                reads_num += 1
                
                # Extract read_id
                read_id_match = re.search(r'>(\S+)', line)
                if not read_id_match:
                    print(f"Warning: Could not extract read_id from line: {line}")
                    i += 1
                    continue
                    
                read_id = read_id_match.group(1)
                
                # Try to extract read_length, if not present set to None
                read_length = None
                read_length_match = re.search(r'(?:len=|Len=|LEN=|L=|l=)(\d+)', line)
                if read_length_match:
                    read_length = read_length_match.group(1)
                
                # Collect sequence data
                sequence_parts = []
                j = i + 1
                while j < len(lines) and not lines[j].startswith('>'):
                    sequence_parts.append(lines[j].replace(' ', ''))
                    j += 1
                
                # Calculate sequence length (if not present in original file)
                # sequence = ''.join(sequence_parts).lower()
                sequence = ''.join(sequence_parts).upper()  # Convert to uppercase
                calculated_length = len(sequence)
                
                # Determine which length value to use
                final_length = read_length if read_length is not None else calculated_length
                
                # Create output file
                read_file_name = f"{genome_name}__{read_id}__{final_length}.txt"
                read_file_path = os.path.join(folder_name, read_file_name)
                
                # Write to file
                with open(read_file_path, 'w') as out_file:
                    out_file.write(sequence)
                
                # Print info, show if calculated length was used
                length_source = "calculated" if read_length is None else "from header"
                print(f"OK, EXTRACTED, reads_num: {reads_num}, read_id: {read_id}, "
                      f"length: {final_length} ({length_source})")
                
                i = j
            else:
                i += 1
        
        print(f"OK, {genome_name} total reads_num: {reads_num}")
        
    except Exception as e:
        print(f"Error processing file {f1_file_path}: {str(e)}")


if __name__ == "__main__":
    file_paths = [
        "~/01_fasta_data/genome_ID1.fasta",
        "~/01_fasta_data/genome_ID2.fasta",
    ]

    output_path = "~/01_fasta_data/01_reads_output"

    for file_path in file_paths:
        process_genome_file(file_path, output_path)
