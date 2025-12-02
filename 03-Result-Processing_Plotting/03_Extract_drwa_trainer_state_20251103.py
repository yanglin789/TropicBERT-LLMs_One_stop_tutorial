# Function: Extract and visualize training metrics from trainer_state.json files
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


####### Extract trainer_state.json file section ##############################################################
def clean_string(value):
    """Clean special characters from strings"""
    if isinstance(value, str):
        return value.replace('\t', ' ').replace('\n', ' ').strip()
    return value


def extract_relevant_data(file_path, output_file_train, output_file_eval):
    """
    Extract relevant data from JSON file and save to two different CSV files (with eval and without eval).
    - If a `log_history` entry **does not contain `eval`** (including keys and values), it belongs to the `train` section.
    - If a `log_history` entry **contains `eval`** (including keys and values), it belongs to the `eval` section.

    :param file_path: Input JSON file path
    :param output_file_train: Output CSV file path for non-eval data
    :param output_file_eval: Output CSV file path for eval data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Cannot parse file {file_path}, file format may be incorrect.")
        return

    log_history = data.get('log_history', [])

    extracted_train_data = []
    extracted_eval_data = []

    for entry in log_history:
        # Check if the entire entry contains 'eval' (including keys and values)
        entry_str = str(entry).lower()  # Convert to string and ignore case
        has_eval = 'eval' in entry_str  # Check if contains 'eval'

        # Clean values in all entries
        cleaned_entry = {key: clean_string(value) for key, value in entry.items()}

        if not has_eval:
            # If does not contain 'eval', assign to train data
            extracted_train_data.append(cleaned_entry)
        else:
            # If contains 'eval', assign to eval data
            extracted_eval_data.append(cleaned_entry)

    if not extracted_train_data and not extracted_eval_data:
        print(f"No relevant data extracted from file {file_path}.")
        return

    # Convert to Pandas DataFrame and save
    try:
        if extracted_train_data:
            df_train = pd.DataFrame(extracted_train_data)
            # Ensure reasonable column order: step and epoch at the front
            base_cols = ['step', 'epoch']
            other_cols = [col for col in df_train.columns if col not in base_cols]
            df_train = df_train[base_cols + other_cols]
            df_train.to_csv(output_file_train, index=False, sep=',', encoding='utf-8')
            print(f"Training data saved to {output_file_train}, containing columns: {list(df_train.columns)}")

        if extracted_eval_data:
            df_eval = pd.DataFrame(extracted_eval_data)
            # Ensure reasonable column order: step and epoch at the front
            base_cols = ['step', 'epoch']
            other_cols = [col for col in df_eval.columns if col not in base_cols]
            df_eval = df_eval[base_cols + other_cols]
            df_eval.to_csv(output_file_eval, index=False, sep=',', encoding='utf-8')
            print(f"Evaluation data saved to {output_file_eval}, containing columns: {list(df_eval.columns)}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


########################################################################################################################

#### Draw each scatter plot section ##############################################################
def plot_curves(train_csv, output_file='training_curves.png', x_axis='epoch'):
    """
    Draw training curves based on CSV file, all metrics plotted in multiple subplots of one image.

    :param train_csv: Training data CSV file path
    :param output_file: Output image file path (PNG format)
    :param x_axis: X-axis selection, 'epoch' or 'step'
    """
    if not os.path.exists(train_csv):
        print(f"File {train_csv} does not exist")
        return

    try:
        # Read data
        df = pd.read_csv(train_csv, sep=',', encoding='utf-8')

        # Check if x-axis column exists
        if x_axis not in df.columns:
            print(f"Specified x-axis '{x_axis}' does not exist in data, will use default 'epoch'")
            x_axis = 'epoch' if 'epoch' in df.columns else 'step'

        # Get all plottable metric columns (excluding step and epoch)
        columns = [col for col in df.columns if col not in ['step', 'epoch']]
        if not columns:
            print("No plottable metric columns found")
            return

        # Set global style
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'font.size': 14,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'grid.color': 'grey',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
        })

        # Create main title (using filename)
        title = os.path.splitext(os.path.basename(train_csv))[0]

        # Create subplot layout - modify to 3 subplots per row
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols  # Calculate rows with ceiling

        # Create figure, using constrained_layout for better control
        fig = plt.figure(figsize=(18, 5 * n_rows), constrained_layout=True)

        # Add global title and explicitly set position
        fig.suptitle(title, fontsize=18, weight='bold', y=1.1)

        # Create GridSpec layout
        gs = GridSpec(n_rows, n_cols, figure=fig)

        # Define color cycle
        colors = plt.cm.tab20.colors

        # Plot each metric subplot
        for i, col in enumerate(columns):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])

            # Calculate max and min values
            max_val = df[col].max()
            min_val = df[col].min()

            # Plot curve and scatter
            ax.plot(df[x_axis], df[col],
                    label=col,
                    color=colors[i % len(colors)],
                    linestyle='-',
                    linewidth=2.5)
            ax.scatter(df[x_axis], df[col],
                       s=50,
                       color=colors[i % len(colors)],
                       alpha=0.7,
                       edgecolors='w',
                       linewidths=1)

            # Set title and labels (add max and min values in title)
            title_with_stats = f"{col} {max_val:.2f}-{min_val:.2f}"
            ax.set_title(title_with_stats, fontsize=16, weight='bold', pad=20)
            ax.set_xlabel(x_axis.capitalize(), fontsize=14, labelpad=10)
            ax.set_ylabel('Value', fontsize=14, labelpad=10)

            # Optimize grid and borders
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add legend
            ax.legend(frameon=False, fontsize=12)

            # Optimize ticks
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Directly save as PNG (high quality)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training curve chart saved to {output_file}")

    except Exception as e:
        print(f"Error drawing or saving training curve chart: {e}")
    finally:
        plt.close()


def process_directory(input_dir, x_axis='epoch'):
    """
    Process all trainer_state.json files in the specified directory

    :param input_dir: Directory path containing trainer_state.json files
    :param x_axis: X-axis selection, 'epoch' or 'step'
    """
    # Find all trainer_state.json files in directory
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == "trainer_state.json":
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"No trainer_state.json files found in directory {input_dir}")
        return

    print(f"Found {len(json_files)} trainer_state.json files, starting processing...")

    for file_path in json_files:
        print(f"\nProcessing file: {file_path}")

        # Extract basic information from input path to generate output path
        input_dir, input_filename = os.path.split(file_path)
        base_name = os.path.splitext(input_filename)[0]

        # Automatically generate output directory (same as input file directory)
        output_dir = os.path.dirname(file_path)

        # Automatically generate CSV output paths
        train_csv_file = os.path.join(output_dir, f"{base_name}_train_metrics_data.csv")
        eval_csv_file = os.path.join(output_dir, f"{base_name}_eval_metrics_data.csv")
        # Automatically generate image output paths
        train_output_file = os.path.join(output_dir, f"{base_name}_train_metrics_{x_axis}.png")
        eval_output_file = os.path.join(output_dir, f"{base_name}_eval_metrics_{x_axis}.png")

        # Call data processing function
        extract_relevant_data(file_path, train_csv_file, eval_csv_file)

        # Only attempt plotting when CSV files exist
        if os.path.exists(train_csv_file):
            plot_curves(train_csv_file, train_output_file, x_axis)
        if os.path.exists(eval_csv_file):
            plot_curves(eval_csv_file, eval_output_file, x_axis)


if __name__ == "__main__":
    # Input directory path (directory containing multiple trainer_state.json files)
    input_directory = r"/jinxianstor/home/yanglin/100_yanglin/03_tropical_plant/06_fine_result/01_sort_out_result_files"
    x_axis = 'epoch'  # step, epoch

    # Process all files in directory
    process_directory(input_directory, x_axis)



