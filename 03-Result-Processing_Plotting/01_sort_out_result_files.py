# Function: Copy files to target directory based on specified workflow
import os
import shutil
import re


def copy_files_to_target_directory(source_root, target_root):
    """
    Copy files to target directory based on specified workflow

    Args:
        source_root: Root path of source directory to traverse
        target_root: Target root directory path
    """
    # Ensure target root directory exists
    os.makedirs(target_root, exist_ok=True)

    # Traverse first-level subdirectories under source root
    for first_level_dir in os.listdir(source_root):
        first_level_path = os.path.join(source_root, first_level_dir)

        # Ensure it's a first-level subdirectory
        if not os.path.isdir(first_level_path):
            continue

        print(f"Processing first-level subdirectory: {first_level_dir}")

        # Check if required files exist
        required_files = ["test_metrics.json", "test_predictions_results.csv"]
        found_files = []
        for file in required_files:
            file_path = os.path.join(first_level_path, file)
            if os.path.exists(file_path):
                found_files.append(file_path)

        # Skip this directory if not all required files are found
        if len(found_files) != len(required_files):
            print(f"Skipping {first_level_dir} - Not all required files found")
            continue

        # Create same-name first-level subdirectory under target root
        target_first_level = os.path.join(target_root, first_level_dir)
        os.makedirs(target_first_level, exist_ok=True)

        # Copy two required files to target directory
        for file_path in found_files:
            shutil.copy2(file_path, target_first_level)
            print(f"Copied {os.path.basename(file_path)} to {target_first_level}")

        # Find checkpoint-* subdirectories
        checkpoint_dirs = []
        checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')

        for item in os.listdir(first_level_path):
            item_path = os.path.join(first_level_path, item)
            if os.path.isdir(item_path):
                match = checkpoint_pattern.match(item)
                if match:
                    checkpoint_dirs.append((int(match.group(1)), item_path))

        # Skip next step if no checkpoint directories found
        if not checkpoint_dirs:
            print(f"No checkpoint directories found in {first_level_dir}")
            continue

        # Find the checkpoint directory with largest numerical value
        checkpoint_dirs.sort(reverse=True)
        selected_checkpoint = checkpoint_dirs[0][1]
        print(f"Selected checkpoint directory: {os.path.basename(selected_checkpoint)}")

        # Check and copy trainer_state.json
        trainer_state_path = os.path.join(selected_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            target_trainer_state = os.path.join(target_first_level, "trainer_state.json")
            shutil.copy2(trainer_state_path, target_trainer_state)
            print(f"Copied trainer_state.json to {target_first_level}")
        else:
            print(f"Warning: trainer_state.json not found in {os.path.basename(selected_checkpoint)}")


if __name__ == "__main__":
    # Example usage - Replace with actual paths
    source_directory = "../02-finetune_all/01-code-finetune/output_fintue/"  # Replace with actual source directory
    target_directory = "./01_sort_out_result_files"  # Replace with actual target directory

    copy_files_to_target_directory(source_directory, target_directory)