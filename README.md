# Implementation-using-Yolov1
To understand how yolo works, I reimplemented yolov1 step by step.
Because of the limitation of resources to train from scratch so this project just trained on more than 2000 images related to dogs and cats.
# Dataset
- Data is available on kaggle https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/data
# Hyperparameters
Training with the number epochs are 100 (6h) on GPU 1660Ti
learning_rate: 2e-5

# Evaluation 
The model is evaluated based on 0.5AP with 0.95% 
The model now have problem with overfitting to fix this problem we need more data to train
# Results
![image](https://github.com/bi-nguyen/Implementation-using-Yolov1/assets/106424285/b1ecb804-1b5a-42ff-aa83-e72422fefd56)
![Figure_2](https://github.com/bi-nguyen/Implementation-using-Yolov1/assets/106424285/95654e7a-3238-47cf-9424-12d2309226fa)



