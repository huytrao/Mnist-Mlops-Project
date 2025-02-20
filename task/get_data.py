from clearml import Task, Dataset
from sklearn.datasets import fetch_openml
import pandas as pd
from pathlib import Path
PROJECT_NAME = 'Minst-Mlops'
PIPELINE_NAME = 'Minst-Pipeline'

task = Task.init(
    project_name=PROJECT_NAME,
    task_name='get data',
    task_type='data_processing',
    reuse_last_task_id=False
)

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
data_path = Path('/tmp/mnist.csv')
mnist_df.to_csv(data_path, index=False)

print(f"Dataset downloaded to: {data_path}")
print(mnist_df.head())

# Create a ClearML dataset
dataset = Dataset.create(
    dataset_name='raw_mnist_dataset',
    dataset_project=PROJECT_NAME
)
# Add the local files we downloaded earlier
dataset.add_files(data_path)
# Let's add some cool graphs as statistics in the plots section!
dataset.get_logger().report_table(title='MNIST Data', series='head', table_plot=mnist_df.head())
# Finalize and upload the data and labels of the dataset
dataset.finalize(auto_upload=True)

print(f"Created dataset with ID: {dataset.id}")
print(f"Data size: {len(mnist_df)}")