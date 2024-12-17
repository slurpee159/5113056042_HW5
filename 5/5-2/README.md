# 5-2: Handwritten Digit Recognition using Dense NN and CNN

## Overview

This project focuses on solving the **handwritten digit recognition problem** using two deep learning approaches:

1. **Dense Neural Networks (DNN)**
2. **Convolutional Neural Networks (CNN)**

The models are implemented in the following frameworks:

- **TensorFlow/Keras**
- **PyTorch**
- **PyTorch Lightning**

Key training features include:

1. **Batch Normalization** for stable training
2. **Dropout** for regularization
3. **EarlyStopping** to avoid overfitting
4. **Learning Rate Scheduler** for dynamic learning rate adjustment

---

## Question 1

**How can I implement a Dense Neural Network (DNN) for handwritten digit recognition using TensorFlow/Keras?**

### Solution:

We build a Dense Neural Network using **TensorFlow/Keras**. The model includes Dropout and Batch Normalization layers, and EarlyStopping is used to stop training when the model starts to overfit.

#### Code Snippet:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Define the Dense NN model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping, lr_scheduler],
    batch_size=32
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Question 2

**How can I implement a CNN for handwritten digit recognition using PyTorch Lightning with callbacks like EarlyStopping?**

### Solution:

We implement a Convolutional Neural Network (CNN) using **PyTorch Lightning**. The model uses Batch Normalization and Dropout, and we include an **EarlyStopping** callback to monitor the validation loss.

#### Code Snippet:

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_data, val_data = random_split(dataset, [50000, 10000])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Define the CNN model
class CNNModel(pl.LightningModule):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

# Train the model
model = CNNModel()
trainer = pl.Trainer(max_epochs=20, callbacks=[
    pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)
])
trainer.fit(model, train_loader, val_loader)
```

---

## Workflow

### Step 1: Data Preparation
- **TensorFlow/Keras**: Load the MNIST dataset and normalize pixel values to [0, 1].
- **PyTorch Lightning**: Use torchvision transforms to preprocess the MNIST data.

### Step 2: Model Implementation
- Implement **DNN** and **CNN** architectures with Batch Normalization and Dropout.

### Step 3: Add Callbacks
- Use EarlyStopping to stop training when validation loss does not improve.
- Use Learning Rate Scheduler to reduce the learning rate dynamically.

### Step 4: Train and Evaluate
- Train the models using the specified frameworks.
- Evaluate the accuracy on the test set.

---

## Notes
- Use **TensorFlow/Keras** for quick prototyping and **PyTorch Lightning** for clean and scalable implementations.
- Install required libraries:

  ```bash
  pip install tensorflow torch torchvision pytorch-lightning
  ```

---

![image](https://github.com/user-attachments/assets/62a5ff28-9b52-4dbe-b274-1eb0f4d5cd17)



