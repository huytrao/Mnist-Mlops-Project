from pathlib import Path
import pandas as pd
from pandasql import sqldf
from datetime import datetime, timedelta
from sklearn.datasets import fetch_openml

def query_database_to_df(query='SELECT * FROM mnist'):
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist['data']
    target = mnist['target']

    # Create a dataframe as mock for the database
    mnist_df = pd.DataFrame(data)
    mnist_df['target'] = target

    # Query the df base on the argument
    mnist_df = sqldf(query, locals())

    # Save resulting DF to disk so it can be added to a clearml dataset as a file
    out_path = Path('/tmp/mnist.csv')
    mnist_df.to_csv(out_path)

    return mnist_df, out_path