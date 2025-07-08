import matplotlib.pyplot as plt
from torchvision import datasets, transforms


#Uncomment the following code to visualize the CIFAR-10 dataset
'''
#Load Dataset
tranform = transforms.ToTensor()
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tranform)

#Visualize Dataset
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    image, label = train_dataset[i]
    axes[i].imshow(image.permute(1, 2, 0))  # Convert from CHW to HWC format
    axes[i].axis('off')
    axes[i].set_title(f'Label: {label}')
plt.show()

#Display Pixel Values of the First Image
image, label = train_dataset[0]
print(f"Image Label: {label}")
print(f"Image shape: {image.shape}")
print("Pixel Values")
print(image)
'''

#Define a simple CNN model using TesorFlow
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

print(model.summary())
print("tensorflow CNN model is ready...")

#Define a simple CNN model using PyTorch
import torch.nn as nn

#Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * 15 * 15, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x= self.pool(x)
        x = x.view(-1, 32 * 15 * 15)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

print(SimpleCNN())
print("PyTorch CNN model is ready...")