CNN Object Detection using TensorFlow & Keras
This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is trained to classify images into 10 different categories.
> FEATURES
CNN Architecture – Multiple convolutional, pooling, and dense layers
Dataset – Uses the CIFAR-10 dataset, which contains 60,000 images across 10 classes
Image Preprocessing – Normalization & One-Hot Encoding
Model Training & Evaluation – Uses EarlyStopping & ModelCheckpoint callbacks
Prediction & Visualization – Displays sample test images with predicted labels
> Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
This project uses the CIFAR-10 dataset, which is automatically downloaded using:
"from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()"
