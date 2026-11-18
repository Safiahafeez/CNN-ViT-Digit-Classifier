import numpy as np
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from models import CNN,ViT
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from preprocessing import prepare_image
import streamlit as st
from PIL import Image, UnidentifiedImageError


## STEP 1 Dataset + split

#Creating output folder
output_folder = Path("module4_outputs")
output_folder.mkdir(exist_ok=True)

#Loading handwritten digit dataset
digits = load_digits()

X = digits.images
y = digits.target

#Normalizing pixel values from 0-16 to 0-1
X = X.astype(np.float32) / 16.0

#Creating indices for shared data splitting
indices = np.arange(len(X))

#Splitting into training and temporary sets
train_val_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#Splitting training and temporary sets into train and validation
train_indices, val_indices = train_test_split(
    train_val_indices,
    test_size=0.25,
    random_state=42,
    stratify=y[train_val_indices]
)

#Saving shared split indices
np.savez(
    output_folder / "split_indices.npz",
    train_indices=train_indices,
    val_indices=val_indices,
    test_indices=test_indices
)

#Displaying dataset information
print("Dataset loaded successfully!")
print("Total images:", len(X))
print("Image shape:", X.shape)
print("Labels:", np.unique(y))
print("Pixel range:", X.min(), "to", X.max())

print("\nDataset split:")
print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Testing samples:", len(test_indices))

print("\nTraining image shape:", X[train_indices].shape)
print("Validation image shape:", X[val_indices].shape)
print("Testing image shape:", X[test_indices].shape)


#Updating
#Setting random seed
torch.manual_seed(42)

#Selecting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

#Loading the saved shared split
saved_splits = np.load(output_folder / "split_indices.npz")

train_indices = saved_splits["train_indices"]
val_indices = saved_splits["val_indices"]

#Preparing CNN input: [N, 1, 8, 8]
X_cnn = X[:, np.newaxis, :, :]

#Converting data into PyTorch tensors
X_tensor = torch.tensor(X_cnn, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

#Creating training and validation datasets
train_dataset = TensorDataset(
    X_tensor[train_indices],
    y_tensor[train_indices]
)

val_dataset = TensorDataset(
    X_tensor[val_indices],
    y_tensor[val_indices]
)

#Creating DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)


## STEP 2 (CNN MODEL)
#Building CNN model
cnn_model = CNN().to(device)

#Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

#Training settings
epochs = 30
best_val_loss = float("inf")

print("\nStarting CNN training...")

for epoch in range(epochs):
    cnn_model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    #Validation phase
    cnn_model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = cnn_model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    val_loss = total_val_loss / total_val
    val_accuracy = correct_val / total_val

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "model_state_dict": cnn_model.state_dict(),
                "num_classes": 10,
                "input_shape": [1, 8, 8],
                "class_names": list(range(10))
            },
            output_folder / "cnn_model.pth"
        )

print("\nCNN training completed!")
print("Best validation loss:", best_val_loss)
print("CNN model saved successfully!")


## STEP 3 (VISION TRANSFORMER TRAINING) (ViT)
#Building ViT model
vit_model = ViT().to(device)

#Loss function and optimizer
vit_criterion = nn.CrossEntropyLoss()
vit_optimizer = torch.optim.Adam(
    vit_model.parameters(),
    lr=0.001
)
#Training settings
vit_epochs = 40
best_vit_val_loss = float("inf")
print("\nStarting ViT training...")
for epoch in range(vit_epochs):

    #Training phase
    vit_model.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        vit_optimizer.zero_grad()

        outputs = vit_model(images)
        loss = vit_criterion(outputs, labels)

        loss.backward()
        vit_optimizer.step()

        total_train_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    #Validation phase
    vit_model.eval()

    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = vit_model(images)
            loss = vit_criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    val_loss = total_val_loss / total_val
    val_accuracy = correct_val / total_val

    print(
        f"ViT Epoch [{epoch + 1}/{vit_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    #Saving the best ViT using validation loss
    if val_loss < best_vit_val_loss:
        best_vit_val_loss = val_loss
        torch.save(
            {
                "model_state_dict": vit_model.state_dict(),
                "num_classes": 10,
                "input_shape": [1, 8, 8],
                "patch_size": 2,
                "num_patches": 16,
                "embedding_dim": 32,
                "class_names": list(range(10))
            },
            output_folder / "vit_model.pth"
        )

print("\nViT training completed!")
print("Best ViT validation loss:", best_vit_val_loss)
print("ViT model saved successfully!")


## STEP 4 (Fair evaluation)
print("\n========== MODEL EVALUATION ==========")

# Loading the saved best CNN checkpoint
cnn_checkpoint = torch.load(
    output_folder / "cnn_model.pth",
    map_location=device,
    weights_only=True
)
cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
cnn_model.eval()

#Loading the saved best ViT checkpoint
vit_checkpoint = torch.load(
    output_folder / "vit_model.pth",
    map_location=device,
    weights_only=True
)
vit_model.load_state_dict(vit_checkpoint["model_state_dict"])
vit_model.eval()

#Loading the untouched test indices
test_indices = saved_splits["test_indices"]

#Preparing test dataset
test_dataset = TensorDataset(
    X_tensor[test_indices],
    y_tensor[test_indices]
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

#Collecting actual labels and model predictions
y_true = []
cnn_predictions = []
vit_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        #CNN predictions
        cnn_outputs = cnn_model(images)
        cnn_preds = cnn_outputs.argmax(dim=1)

        #ViT predictions
        vit_outputs = vit_model(images)
        vit_preds = vit_outputs.argmax(dim=1)

        y_true.extend(labels.numpy())
        cnn_predictions.extend(cnn_preds.cpu().numpy())
        vit_predictions.extend(vit_preds.cpu().numpy())

#Converting predictions to NumPy arrays
y_true = np.array(y_true)
cnn_predictions = np.array(cnn_predictions)
vit_predictions = np.array(vit_predictions)

#MAJORITY-CLASS BASELINE

#Finding majority class using training labels only
majority_class = np.bincount(
    y_tensor[train_indices].numpy()
).argmax()
baseline_predictions = np.full(
    len(y_true),
    majority_class
)
baseline_accuracy = accuracy_score(
    y_true,
    baseline_predictions
)
baseline_f1 = f1_score(
    y_true,
    baseline_predictions,
    average="macro",
    zero_division=0
)
print("\nMAJORITY-CLASS BASELINE")
print("Majority class:", majority_class)
print("Test Accuracy:", round(baseline_accuracy, 4))
print("Macro F1:", round(baseline_f1, 4))

#CNN EVALUATION
cnn_accuracy = accuracy_score(
    y_true,
    cnn_predictions
)
cnn_f1 = f1_score(
    y_true,
    cnn_predictions,
    average="macro"
)
cnn_cm = confusion_matrix(
    y_true,
    cnn_predictions,
    labels=np.arange(10)
)
print("\nCNN RESULTS")
print("Test Accuracy:", round(cnn_accuracy, 4))
print("Macro F1:", round(cnn_f1, 4))
print("Confusion Matrix:")
print(cnn_cm)

#ViT EVALUATION
vit_accuracy = accuracy_score(
    y_true,
    vit_predictions
)
vit_f1 = f1_score(
    y_true,
    vit_predictions,
    average="macro"
)

vit_cm = confusion_matrix(
    y_true,
    vit_predictions,
    labels=np.arange(10)
)

print("\nVIT RESULTS")
print("Test Accuracy:", round(vit_accuracy, 4))
print("Macro F1:", round(vit_f1, 4))
print("Confusion Matrix:")
print(vit_cm)

#IDENTIFYING MODEL MISTAKES
def show_mistakes(actual, predicted, model_name):

    print(f"\n{model_name} - Example mistakes:")

    wrong_indices = np.where(actual != predicted)[0]

    if len(wrong_indices) == 0:
        print("No incorrect predictions.")
        return

    for index in wrong_indices[:2]:

        print(
            f"Actual digit: {actual[index]}, "
            f"Predicted digit: {predicted[index]}"
        )

show_mistakes(y_true, cnn_predictions, "CNN")
show_mistakes(y_true, vit_predictions, "ViT")

#SAVING RESULTS
results = {
    "majority_class": int(majority_class),
    "baseline_accuracy": float(baseline_accuracy),
    "baseline_macro_f1": float(baseline_f1),
    "cnn_accuracy": float(cnn_accuracy),
    "cnn_macro_f1": float(cnn_f1),
    "vit_accuracy": float(vit_accuracy),
    "vit_macro_f1": float(vit_f1)
}
with open(output_folder / "evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

#Saving confusion matrices
np.save(
    output_folder / "cnn_confusion_matrix.npy",
    cnn_cm
)

np.save(
    output_folder / "vit_confusion_matrix.npy",
    vit_cm
)

print("\nEvaluation results saved successfully!")
print("CNN confusion matrix saved!")
print("ViT confusion matrix saved!")

show_mistakes(y_true, cnn_predictions, "CNN")
show_mistakes(y_true, vit_predictions, "ViT")

#Finding the most common digit confusions
def interpret_mistakes(cm, model_name):

    cm_without_diagonal = cm.copy()
    np.fill_diagonal(cm_without_diagonal, 0)
    actual_digit, predicted_digit = np.unravel_index(
        np.argmax(cm_without_diagonal),
        cm_without_diagonal.shape
    )
    mistakes = cm_without_diagonal[actual_digit, predicted_digit]
    print(f"\n{model_name} mistake interpretation:")

    if mistakes == 0:
        print("No incorrect classifications on the test set.")
    else:
        print(
            f"The model most often confused digit {actual_digit} "
            f"with digit {predicted_digit} ({mistakes} times)."
        )
interpret_mistakes(cnn_cm, "CNN")
interpret_mistakes(vit_cm, "ViT")



##STREAMLIT APP

#Setting the device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Setting the output folder
output_folder = Path(__file__).parent / "module4_outputs"

#Loading the saved CNN model
cnn_checkpoint = torch.load(
    output_folder / "cnn_model.pth",
    map_location=device,
    weights_only=True
)
cnn_model = CNN()
cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
cnn_model.to(device)
cnn_model.eval()

#Loading the saved ViT model
vit_checkpoint = torch.load(
    output_folder / "vit_model.pth",
    map_location=device,
    weights_only=True
)
vit_model = ViT()
vit_model.load_state_dict(vit_checkpoint["model_state_dict"])
vit_model.to(device)
vit_model.eval()


