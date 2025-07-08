# CNN Model Setup with TensorFlow and PyTorch

This project demonstrates the setup of a simple Convolutional Neural Network (CNN) using two popular deep learning frameworks: TensorFlow and PyTorch. It includes a commented-out section for visualizing the CIFAR-10 dataset, which is a common dataset used for image classification tasks.

## Project Structure

The core of this project consists of Python code snippets that define CNN models in both TensorFlow and PyTorch.

### 1. Dataset Visualization (Commented Out)

The initial section of the code, currently commented out, provides a utility to:

* **Load Dataset**: Download and load the CIFAR-10 dataset using `torchvision.datasets`.
* **Visualize Dataset**: Display the first few images from the training dataset, along with their corresponding labels, using `matplotlib`. This helps in understanding the nature of the data.
* **Display Pixel Values**: Print the shape and raw pixel values of the first image to illustrate the numerical representation of image data.

To enable this visualization, simply uncomment the relevant lines of code.

### 2. TensorFlow CNN Model Setup

A simple CNN model is defined using TensorFlow's Keras API. This model is a sequential stack of layers designed for image classification.

**Model Architecture:**

* **`tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3))`**:
    * A 2D convolutional layer with 32 filters, each of size 3x3.
    * `activation='relu'` applies the Rectified Linear Unit activation function.
    * `input_shape=(32, 32, 3)` specifies the expected input shape of the images (height, width, color channels for RGB).
* **`tf.keras.layers.MaxPooling2D((2, 2))`**:
    * A max pooling layer with a 2x2 pool size, which reduces the spatial dimensions of the feature maps by taking the maximum value within each window.
* **`tf.keras.layers.Flatten()`**:
    * Flattens the output from the previous layer into a 1D vector, preparing it for the fully connected layers.
* **`tf.keras.layers.Dense(128, activation='relu')`**:
    * A fully connected (dense) layer with 128 neurons and ReLU activation.
* **`tf.keras.layers.Dense(10, activation='softmax')`**:
    * The output layer with 10 neurons (corresponding to the 10 classes in CIFAR-10) and a `softmax` activation function, which outputs probability distributions over the classes.

The `model.summary()` method provides a concise overview of the model's layers, output shapes, and the number of trainable parameters.

### 3. PyTorch CNN Model Setup

A simple CNN model is defined using PyTorch's `torch.nn` module. This model is implemented as a Python class inheriting from `nn.Module`.

**Model Architecture:**

* **`self.conv1 = nn.Conv2d(3, 32, kernel_size=3)`**:
    * A 2D convolutional layer.
    * `3` is the number of input channels (for RGB images).
    * `32` is the number of output channels (filters).
    * `kernel_size=3` specifies a 3x3 convolution kernel.
* **`self.pool = nn.MaxPool2d(2,2)`**:
    * A max pooling layer with a 2x2 pool size.
* **`self.fc1 = nn.Linear(32 * 15 * 15, 128)`**:
    * The first fully connected (linear) layer. The input size `32 * 15 * 15` is calculated based on the output size of the preceding convolutional and pooling layers assuming a 32x32 input image and a 3x3 kernel with stride 1 and no padding for `conv1`, followed by a 2x2 max pool.
    * `128` is the number of output features.
* **`self.fc2 = nn.Linear(128, 10)`**:
    * The second fully connected (output) layer with 10 output features (for 10 classes).

**`forward(self, x)` method:**

This method defines the forward pass of the network, specifying how input data `x` flows through the layers:

1.  `x = F.relu(self.conv1(x))`: Applies convolution followed by ReLU activation.
2.  `x = self.pool(x)`: Applies max pooling.
3.  `x = x.view(-1, 32 * 15 * 15)`: Flattens the tensor before passing it to the fully connected layers. `-1` infers the batch size, and `32 * 15 * 15` is the calculated size of the flattened features.
4.  `x = F.relu(self.fc1(x))`: Applies the first fully connected layer followed by ReLU.
5.  `x = self.fc2(x)`: Applies the final fully connected layer.

The `print(SimpleCNN())` statement will output the structure of the PyTorch model.

## Prerequisites

To run this code, you will need:

* Python 3.x
* `tensorflow`
* `torch`
* `torchvision`
* `matplotlib` (for visualization)
* `numpy` (often a dependency, good to have)

You can install these libraries using pip:

```bash
pip install tensorflow torch torchvision matplotlib numpy