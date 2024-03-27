import sys
import asyncio
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert import HTMLExporter
import os
import zipfile
import time

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def parse_directory(subdir_path, base_path, country, focus_country):
    # Strip the base_path and the model directory to get the target subdirectory
    target_subdir = subdir_path.replace(base_path, '').strip(os.sep).split(os.sep)[0]

    # Split the remaining path to get the model info
    remaining_path = subdir_path.replace(os.path.join(base_path, target_subdir), '').strip(os.sep)
    parts = remaining_path.split(os.sep)
    model_info = parts[0]

    if model_info:
        model = model_info.split('_')[0]
        user_model = model_info.split(f'controlGroup{control_country}_')[-1]
        path = os.path.join(base_path, target_subdir, model_info)
        odataset_path = os.path.join(path, f'{prefix}_' + model.lower())
        filenames = f'{prefix}_' + model.lower()
        date_from_run = model_info.split('_')[2]
        return model, user_model, country, focus_country, path, odataset_path, filenames, target_subdir, date_from_run
    else:
        return None, None, None, None, None, None, None, None, None

def extract_zip_if_needed(_path):
    # Extract the relevant part of the path
    category = '/'.join(_path.split('/')[:3])  # Extracts 'data/ChooseSome_ControlGroupDE/Only US Tracks chosen/'
    if os.path.exists(category):
        for filename in os.listdir(category):
            if filename.endswith('.zip'):
                # Construct the full file path
                file_path = os.path.join(category, filename)

                # Construct the directory to unzip to (same name as the zip file, but without .zip)
                extract_path = os.path.join(category, filename[:-4])

                # Create the directory if it doesn't exist
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                time.sleep(1)

                os.remove(file_path)
    else:
        print(f"The path {category} does not exist.")

def process_directories(parent_directory, html_export_path, notebook_path, country, focus_country):
    subdirectories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory)
                      if os.path.isdir(os.path.join(parent_directory, d))]

    # Loop through each subdirectory and process
    for subdir_path in subdirectories:
        print(f"Processing directory: {subdir_path}")
        extract_zip_if_needed(subdir_path)
        # Now process the second level of directories
        second_level_dirs = [os.path.join(subdir_path, d) for d in os.listdir(subdir_path)
                             if os.path.isdir(os.path.join(subdir_path, d))]

        for second_level_dir in second_level_dirs:
            print(f"Processing subdirectory: {second_level_dir}")
            variables = parse_directory(second_level_dir, parent_directory, country, focus_country)

            if variables[0] is not None:
                nb = run_notebook(notebook_path, variables)
                convert_to_html(nb, variables, html_export_path)
            else:
                print(f"Failed to parse subdirectory for required variables: {second_level_dir}")


def run_notebook(notebook_path, variables):
    print("Opening notebook...")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Unpack variables
    model, user_model, country, focus_country, path, odataset_path, filenames, target_subdir, date_from_run = variables
    # Fix the paths using raw string literals or by replacing single backslashes with double backslashes
    path = path.replace('\\', '\\\\')
    odataset_path = odataset_path.replace('\\', '\\\\')
    # Create a code string that defines these variables using raw string literals
    variable_code = f"""
    model = '{model}'
    user_model = '{user_model}'
    country = '{country}'
    focus_country = '{focus_country}'
    path = r'{path}'
    odataset_path = r'{odataset_path}'
    filenames = '{filenames}'
    control_country = '{control_country}'
    """

    # Insert the variable definition code as the first cell
    nb.cells.insert(0, nbformat.v4.new_code_cell(variable_code))

    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except CellExecutionError as e:
        print("Error executing the notebook. See above for error.")
        raise e
    print("Notebook execution completed.")
    return nb


def convert_to_html(nb, variables, base_html_export_path):
    model, user_model, country, focus_country, path, odataset_path, filenames, target_subdir, date_from_run = variables

    # Normalize the subdirectory path
    target_subdir = target_subdir.replace('\\', '/')

    # Construct the full export path using the base path and the target subdirectory
    full_export_path = os.path.join(base_html_export_path, target_subdir)

    # Create the subdirectory if it doesn't exist
    os.makedirs(full_export_path, exist_ok=True)

    # Construct the full path for the HTML file
    html_filename = f'plots_{date_from_run}_{model}_{user_model}_{country}_{focus_country}.html'
    full_html_path = os.path.join(full_export_path, html_filename)

    print(
        f"Converting notebook to HTML. Model: {model}, User Model: {user_model}, Country: {country}, Focus Country: {focus_country}...")
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)

    # Save the HTML file
    with open(full_html_path, 'w', encoding='utf-8') as f:
        f.write(body)
    print(f"Notebook successfully converted to HTML: {full_html_path}\n")


# Paths
notebook_path = 'creating_plots_new.ipynb'
html_export_path = 'plots/ChooseSome_ControlGroupDE/'
base_path = 'data/ChooseSome_ControlGroupDE/'
prefix = 'smm_demo'
country = 'DE'
focus_country = 'US'
control_country = 'DE'

print("Starting the process...")
process_directories(base_path, html_export_path, notebook_path, country, focus_country)
print("Process completed.")
country = 'FI'
print("Starting the process...")
process_directories(base_path, html_export_path, notebook_path, country, focus_country)
print("Process completed.")

html_export_path = 'plots/ChooseSome_ControlGroupDE_sa2se3/'
base_path = 'data/ChooseSome_ControlGroupDE_sa2se3/'
prefix = 'smm_demo2'
country = 'DE'

print("Starting the process...")
process_directories(base_path, html_export_path, notebook_path, country, focus_country)
print("Process completed.")

country = 'FI'

print("Starting the process...")
process_directories(base_path, html_export_path, notebook_path, country, focus_country)
print("Process completed.")