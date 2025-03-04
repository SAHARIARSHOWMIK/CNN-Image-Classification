
# Convolutional Neural Networks – Image Classification

This project aims to perform supervised image classification on a set of colored images using a convolutional neural network (CNN) architecture. It leverages data augmentation and transformations to accurately assign images to one of 10 predefined categories.

## Dataset (CIFAR-10)
The project uses the CIFAR-10 dataset—a standard benchmark in computer vision and machine learning for image classification tasks. Key characteristics include:
- **Total Images:** 60,000 32x32 color images, with 6,000 images per class.
- **Splits:** 50,000 images for training and 10,000 images for testing.
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Image Classification Workflow
The project is organized into several steps that mimic the typical stages of data processing and analysis.

### **CIFAR-10 Classification Using TensorFlow**

1. **Initialization:** Import all necessary libraries and modules.
2. **Loading the Dataset:** Retrieve the CIFAR-10 dataset from the Keras library in TensorFlow and examine its details.
3. **Image Preprocessing:** Enhance and augment the data using TensorFlow’s ImageDataGenerator by:
   - Scaling image pixel values to the [0, 1] range.
   - Applying random shear transformations.
   - Applying random zoom transformations.
   - Flipping images horizontally at random.
4. **Building the CNN Model:** Construct a Sequential CNN with:
   - An input layer.
   - Two convolutional layers (with ReLU activation) that gradually increase the number of filters.
   - Two max pooling layers following the convolutional layers.
   - A flattening layer to prepare data for the dense layers.
   - Two dense (fully connected) layers with ReLU activation.
   - An output layer with Softmax activation to classify the images.
5. **Model Training:** Compile and train the model with the following settings:
   - **Optimizer:** Adam
   - **Loss Function:** Categorical Crossentropy
   - **Batch Size:** 32
   - **Epochs:** 25
6. **Performance Analysis:** Evaluate model performance by plotting training and validation accuracy over the epochs.


### **CIFAR-10 Classification Using PyTorch**


1. **Initialization:** Import the required libraries and modules.
2. **Loading and Transforming the Dataset:** 
   - Load the CIFAR-10 dataset using DataLoader from the torchvision library with a batch size of 32 and shuffling enabled.
   - Use Compose to apply data augmentations such as:
     - Random rotations.
     - Random horizontal flips.
     - Color jitter (to adjust brightness, contrast, saturation, and hue).
     - Scaling pixel values to the [0, 1] range.
3. **Building the CNN Model:** Define a CNN model using nn.Module that includes:
   - An input layer.
   - Two convolutional layers with ReLU activation and increasing filter counts.
   - Two max pooling layers.
   - A flattening layer.
   - Two dense layers with ReLU activation.
   - An output layer with Softmax activation.
   - Configure the model with the Adam optimizer and CrossEntropyLoss.
4. **Model Training:** Train the model for 25 epochs.
5. **Performance Analysis:** Plot and analyze the training and validation accuracy over epochs.



