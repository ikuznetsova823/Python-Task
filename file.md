# VIN Character Recognition Documentation

This documentation provides an overview of the VIN Character Recognition project, including the used data, methods, ideas, accuracy report, usage instructions, and author information.

## Used Data

The VIN Character Recognition project utilizes the following datasets:

1. EMNIST Dataset: The EMNIST dataset is an extended version of the classic MNIST dataset. It contains handwritten characters, including both alphabets (A-Z) and numbers (0-9). The dataset was preprocessed to include only the characters present in the VIN code, resulting in a 33-class dataset.

## Methods and Ideas

The VIN Character Recognition project uses a Convolutional Neural Network (CNN) architecture for character classification. The CNN model consists of multiple layers, including convolutional layers, pooling layers, and dense layers. The architecture is designed to learn and classify handwritten characters based on their visual features.

The CNN model is trained on the EMNIST dataset using the Adam optimizer and the sparse categorical cross-entropy loss function. The model is trained for a specified number of epochs to optimize its performance in character recognition.

## Accuracy Report

The accuracy of the VIN Character Recognition model is reported based on its performance on the test dataset. The accuracy metric provides an assessment of how well the model can classify VIN characters correctly.

## Author Information

The VIN Character Recognition project was developed by Iryna. For any inquiries or feedback, please contact me at iryna.kuznezova@gmail.com.
