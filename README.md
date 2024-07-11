# Leaf-Disease-Detection-using-CNN
This project demonstrates a simple implementation of a tomato leaf disease classification system using a Convolutional Neural Network (CNN). The model is designed to identify 10 common tomato diseases and distinguish them from healthy leaves.

Dataset
The project utilizes the Tomato Leaf Disease dataset available on Kaggle: https://www.kaggle.com/datasets/noulam/tomato/data

The dataset consists of:

10 Disease Classes: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus
Healthy Class: Images of disease-free leaves
Images: Original resolution (resized to 64x64 pixels for the model)
Prerequisites
Python (3.x recommended)
Libraries:
pandas
OpenCV (cv2)
NumPy
matplotlib
seaborn
TensorFlow (with Keras)
scikit-learn
Google Colab (optional, but code is written assuming this environment)

How to Run
1)Clone the Repository:
  git clone https://your-github-repository-url.git

2)Install Dependencies:
  pip install -r requirements.txt 

3)Prepare Dataset:
  Download the Tomato Leaf Disease dataset from Kaggle and organize it into "train" and "valid" folders as expected by the     code.
  Adjust file paths in the code (DATASET and DATASET2) to point to your data directory.
  
4)Execute the Code: 
  Load and preprocess the images.
  Train the CNN model.
  Evaluate model performance on the validation set.
  Generate plots of training/validation loss and accuracy.
  Make predictions and analyze results using ROC-AUC curves and a confusion matrix.

Model
Architecture: The model uses a sequential CNN with convolutional, pooling, and fully-connected layers. The architecture is defined within the code.
Training: The model is trained using the Adam optimizer and categorical cross-entropy loss.
Evaluation: Performance is assessed using accuracy, ROC-AUC curves, and a confusion matrix.
