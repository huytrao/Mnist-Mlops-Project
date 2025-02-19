from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from clearml import Dataset, Task
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Connecting ClearML with the current process,
# from here on everything is logged automatically
import global_config

task = Task.init(
    project_name=global_config.PROJECT_NAME,
    task_name='model training',
    output_uri=True
)

# Set default docker
task.set_base_docker(docker_image="python:3.9.13")

# Training args
training_args = {
    'eval_metric': "mlogloss",
    'objective': 'multi:softmax',
    'num_class': 10,  # MNIST có 10 lớp (0-9)
    'test_size': 0.2,
    'random_state': 42,
    'tree_method': 'hist'  # Sử dụng CPU
}
task.connect(training_args)

# Load our Dataset
local_path = Dataset.get(
    dataset_name='preprocessed_mnist_dataset',
    dataset_project=global_config.PROJECT_NAME
).get_local_copy()
local_path = Path(local_path)
# local_path = Path('data/preprocessed_data')
X = pd.read_csv(local_path / 'mnist.csv', index_col=0)
y = X.pop('target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training_args['test_size'], random_state=training_args['random_state'])
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train
bst = xgb.train(
    {
        'eval_metric': training_args['eval_metric'],
        'objective': training_args['objective'],
        'num_class': training_args['num_class'],
        'random_state': training_args['random_state'],
        'tree_method': training_args['tree_method']
    },
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0
)

bst.save_model("best_model")

# Vẽ biểu đồ feature importance
importance = bst.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [f'pixel{i}' for i in range(1, 785)],
    'Importance': [importance.get(f'f{i}', 0) for i in range(784)]
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Vẽ phân phối các giá trị pixel
pixel_values = X.values.flatten()
plt.figure(figsize=(10, 6))
sns.histplot(pixel_values, bins=50, kde=True)
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# Vẽ một số hình ảnh từ dữ liệu MNIST
def plot_mnist_images(df, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
    for i in range(num_images):
        image = df.iloc[i, :-1].values
        if len(image) == 784:  # Đảm bảo rằng mỗi hàng có đủ 784 giá trị pixel
            image = image.reshape(28, 28)
            label = df.iloc[i, -1]
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
    plt.show()

plot_mnist_images(pd.concat([X, y], axis=1), num_images=5)

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average='macro')
print(f"Model trained with accuracy: {accuracy} and recall: {recall}")
# Save the actual accuracy as an artifact so we can get it as part of the pipeline
task.get_logger().report_scalar(
    title='Performance',
    series='Accuracy',
    value=accuracy,
    iteration=0
)
task.get_logger().report_scalar(
    title='Performance',
    series='Recall',
    value=recall,
    iteration=0
)
print("Done")