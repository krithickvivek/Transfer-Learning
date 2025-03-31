# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
The objective is to classify images from a given dataset using the VGG-19 model through transfer learning. The dataset consists of images from multiple classes, organized for use with the ImageFolder module in PyTorch, and is divided into training and testing subsets.

## DESIGN STEPS
### STEP 1:
Load the dataset using ImageFolder and apply transformations such as resizing to 224x224, converting to tensors, and normalizing with VGG-19's mean and standard deviation.

### STEP 2:
Load the pre-trained VGG-19 model, freeze initial convolutional layers to retain learned features, and modify the final fully connected layer to match the number of output classes.

### STEP 3:
Train the model using CrossEntropyLoss and an optimizer like Adam or SGD. Monitor the training loss and accuracy during the process.

### STEP 4:
Evaluate the model on the test dataset and calculate accuracy, precision, recall, and F1-score.

### STEP 5:
Visualize sample predictions, loss curves, and the confusion matrix to analyze the model's performance.


## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
num_classes=1
in_features=model.classifier[-1].in_features
model.classifier[-1]=nn.Linear(in_features,num_classes)

# Include the Loss function and optimizer
criterion= nn.BCEWithLogitsLoss()
optimizer= optim.Adam(model.classifier[-1].parameters(),lr=0.01)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Krithick Vivekananda")
    print("Register Number: 212223240075")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/a54d2770-24b8-498e-81aa-f5cea127bfd2)
### Confusion Matrix
![image](https://github.com/user-attachments/assets/bb2e4f5b-e7f0-4a8e-8dfa-4a6a217514dc)
### Classification Report
![image](https://github.com/user-attachments/assets/ac2293bb-059b-44c7-b3a8-1f00b6771ccd)
### New Sample Prediction
![image](https://github.com/user-attachments/assets/57e86dbd-fa97-41bb-bc8a-7e7b22a41e1a)
![image](https://github.com/user-attachments/assets/ce0545f6-98aa-4119-a6ca-094aadaebfef)

## RESULT
The VGG-19 transfer learning model was successfully implemented and trained. It achieved good classification performance with minimized training and validation losses, accurate predictions, and satisfactory evaluation metrics
