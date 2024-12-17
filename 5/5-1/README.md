## 5-1: Iris Classification with Dropout, Batch Normalization, and EarlyStopping

## Overview

This project focuses on solving the **Iris classification problem** using multiple deep learning frameworks:

- **TensorFlow/Keras**
- **PyTorch**
- **PyTorch Lightning**

The models incorporate key training schemes:

1. **Dropout** for regularization
2. **Batch Normalization** for stable training
3. **EarlyStopping** to avoid overfitting
4. **Learning Rate Scheduler** for dynamic learning rate adjustment

---

## Question 1

**How can I implement an Iris classification model using TensorFlow/Keras with Dropout and EarlyStopping?**

### Solution:

We build a neural network model using TensorFlow/Keras that includes Dropout layers and Batch Normalization. We use the **EarlyStopping** callback to monitor the validation loss and stop training when the model begins to overfit.

#### Code Snippet:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Keras model
model = Sequential([
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## Question 2

**How can I implement the Iris classification model using PyTorch Lightning with EarlyStopping and Learning Rate Scheduler?**

### Solution:

We use PyTorch Lightning to organize the training process. The model includes Batch Normalization and Dropout, and the training loop integrates EarlyStopping and a Learning Rate Scheduler.

#### Code Snippet:

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the PyTorch Lightning model
class IrisModel(pl.LightningModule):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

# Train the model
model = IrisModel()
trainer = pl.Trainer(max_epochs=50, callbacks=[
    pl.callbacks.EarlyStopping(monitor='train_loss', patience=5)
])
trainer.fit(model, train_loader)

# Test the model
trainer.test(dataloaders=test_loader)
```

---

## Workflow

### Step 1: Data Preparation
- Load the Iris dataset and preprocess it using **StandardScaler** for normalization.
- Convert the target variable into a one-hot encoded format for Keras and class labels for PyTorch.

### Step 2: Model Implementation
- **TensorFlow/Keras**: Implement a feedforward neural network with Dropout and Batch Normalization.
- **PyTorch Lightning**: Implement the model with PyTorch, organizing it into a LightningModule for clean training.

### Step 3: Add Callbacks
- Use **EarlyStopping** to halt training when validation loss stops improving.
- Use **Learning Rate Scheduler** to adjust the learning rate dynamically during training.

### Step 4: Training and Evaluation
- Train the models using appropriate optimizers (e.g., Adam) and evaluate their accuracy on the test set.

---

## Notes
- TensorFlow/Keras and PyTorch Lightning both provide excellent tools for building and training deep learning models.
- Use callbacks like EarlyStopping and Learning Rate Scheduler to improve training efficiency and avoid overfitting.
- Ensure the required libraries are installed:
  ```bash
  pip install tensorflow torch pytorch-lightning scikit-learn
  ```

---
## Result

![image](https://github.com/user-attachments/assets/fb79d052-7593-4575-9e2b-3d3146cfd693)



