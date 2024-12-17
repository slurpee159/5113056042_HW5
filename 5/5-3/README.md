# 5-3: CIFAR Image Classification using VGG19 Pretrained Model

## Overview

This project focuses on solving the **CIFAR image classification problem** using a **VGG19 pretrained model**. The implementation leverages two popular frameworks:

- **TensorFlow/Keras**
- **PyTorch Lightning**

Key features include:
1. Using **VGG19 pretrained weights** for transfer learning.
2. Fine-tuning the pretrained model for CIFAR image classification.
3. Incorporating essential training strategies such as Dropout, Batch Normalization, EarlyStopping, and Learning Rate Scheduler.

---

## Question 1

**How can I use the VGG19 pretrained model in TensorFlow/Keras to classify CIFAR-10 images?**

### Solution:

We load the **VGG19 pretrained model** without the top layers, add custom Dense layers for CIFAR-10 classification, and fine-tune the model.

#### Code Snippet:

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Load VGG19 pretrained model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # Freeze the base model

# Add custom classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
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
    epochs=30,
    callbacks=[early_stopping, lr_scheduler],
    batch_size=32
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Question 2

**How can I implement the VGG19 pretrained model for CIFAR-10 classification using PyTorch Lightning?**

### Solution:

We use the **VGG19 model** from `torchvision.models` as the base, modify its fully connected layers for CIFAR-10, and train using PyTorch Lightning.

#### Code Snippet:

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Data preparation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the PyTorch Lightning model
class VGG19CIFAR(pl.LightningModule):
    def __init__(self):
        super(VGG19CIFAR, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 10)  # Modify for CIFAR-10
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
model = VGG19CIFAR()
trainer = pl.Trainer(max_epochs=20, callbacks=[
    pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)
])
trainer.fit(model, train_loader, test_loader)
```

---

## Workflow

### Step 1: Data Preparation
- **TensorFlow/Keras**: Load CIFAR-10 dataset and normalize pixel values.
- **PyTorch Lightning**: Use torchvision datasets to load and preprocess CIFAR-10 images.

### Step 2: VGG19 Model Implementation
- Use VGG19 pretrained weights with `include_top=False` in TensorFlow/Keras.
- Modify the classifier layers in PyTorch Lightning to adapt VGG19 for CIFAR-10.

### Step 3: Add Training Strategies
- **Dropout**: Added to reduce overfitting.
- **Batch Normalization**: Stabilizes training.
- **EarlyStopping**: Stops training when validation loss does not improve.
- **Learning Rate Scheduler**: Dynamically reduces the learning rate.

### Step 4: Training and Evaluation
- Train the models using respective frameworks.
- Evaluate accuracy and performance on the CIFAR-10 test set.

---

## Notes
- Transfer learning with VGG19 significantly improves model performance on CIFAR-10.
- Ensure required libraries are installed:

  ```bash
  pip install tensorflow torch torchvision pytorch-lightning
  ```

---

![image](https://github.com/user-attachments/assets/5015054d-3c3c-4a5a-9a49-3cd309e81fe2)

