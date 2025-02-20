import os.path
from pathlib import Path
import pandas as pd
from clearml import Dataset, Task
from sklearn.datasets import fetch_openml

PROJECT_NAME = 'Minst-Mlops'
PIPELINE_NAME = 'Minst-Pipeline'

task = Task.init(
    project_name=PROJECT_NAME,
    task_name='preprocess data',
    task_type='data_processing',
    reuse_last_task_id=False
)

# Create the folder we'll output the preprocessed data into
preprocessed_data_folder = Path('/tmp')
if not os.path.exists(preprocessed_data_folder):
    os.makedirs(preprocessed_data_folder)

# Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist['data']
target = mnist['target']

# Create a dataframe as mock for the database
mnist_df = pd.DataFrame(data)
mnist_df['target'] = target

# Convert categorical columns to string to avoid fillna issues
for col in mnist_df.select_dtypes(['category']).columns:
    mnist_df[col] = mnist_df[col].astype(str)

# Save resulting DF to disk so it can be added to a clearml dataset as a file
mnist_df.to_csv(preprocessed_data_folder / 'mnist.csv', index=False)

# Create a new version of the dataset, which is cleaned up
new_dataset = Dataset.create(
    dataset_project=PROJECT_NAME,
    dataset_name='preprocessed_mnist_dataset'
)
new_dataset.add_files(preprocessed_data_folder / 'mnist.csv')
new_dataset.get_logger().report_table(title='MNIST data', series='head', table_plot=mnist_df.head())
new_dataset.finalize(auto_upload=True)

# Log to console which dataset ID was created
print(f"Created preprocessed dataset with ID: {new_dataset.id}")