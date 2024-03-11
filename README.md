# Implementation-using-Yolov1
To understand how yolo works, I reimplemented yolov1 step by step
Because of the limitation of resources to train from scratch, In this project, I just trained on more than 2000 images related to dogs and cats.
# Dataset
- Data is available on kaggle https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/data
# Hyperparameters
Training with the number epochs are 100 (6h) on GPU 1660Ti
learning_rate: 2e-5

# Evaluation 
The model is evaluated based on 0.5AP with 0.95% 
The model now have problem with overfitting to fix this problem we need more data to train



