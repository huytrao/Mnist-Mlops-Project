from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        df = pd.DataFrame(columns=body.keys())
        df.loc[0] = body.values()
        
        # Chọn các cột pixel từ dữ liệu MNIST
        pixel_columns = [f'pixel{i}' for i in range(1, 785)]
        X = df[pixel_columns]
        
        return xgb.DMatrix(X)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=round(data[0]), y_raw=float(data[0]))

    def plot_mnist_images(self, df: pd.DataFrame, num_images: int = 5):
        """Vẽ một số hình ảnh từ dữ liệu MNIST"""
        fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
        for i in range(num_images):
            image = df.iloc[i, :-1].values.reshape(28, 28)
            label = df.iloc[i, -1]
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.show()

    def plot_pixel_distribution(self, df: pd.DataFrame):
        """Vẽ phân phối các giá trị pixel trong dữ liệu MNIST"""
        pixel_values = df.iloc[:, :-1].values.flatten()
        plt.figure(figsize=(10, 6))
        sns.histplot(pixel_values, bins=50, kde=True)
        plt.title('Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()


# Ví dụ sử dụng các hàm vẽ biểu đồ
if __name__ == "__main__":
    # Giả sử bạn đã có DataFrame mnist_df từ dữ liệu MNIST
    mnist_df = pd.read_csv('/tmp/mnist.csv')
    
    preprocess = Preprocess()
    preprocess.plot_mnist_images(mnist_df, num_images=5)
    preprocess.plot_pixel_distribution(mnist_df)