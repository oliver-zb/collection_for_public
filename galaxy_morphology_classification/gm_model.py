import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import random
import numpy as np


def set_seed(seed=42):
    """set seeds for all relevant libraries to ensure reproducibility of results across runs for model performance comparison"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed {seed} gesetzt für Reproduzierbarkeit", flush=True)

from gm_datapipeline import load_galaxy_data, get_galaxy_batch

#function with 32 bit precision (float32) for training the model.
def train_model(ml_data, epochs=10, batch_size=32, learning_rate=0.001, val_split=0.15, test_split=0.15, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images, labels = ml_data
    num_classes = len(labels[0])

    # convert data to tensors
    dataset = TensorDataset(images, labels)

    # Train/Val/Test Split
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # mum workers seems not to help a lot
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # use the pre-trained resnet18 model and modify the final layer for our number of classes
    model = models.resnet18(weights="IMAGENET1K_V1")
    # change the last fully connected layer to match the number of classes in our dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # shift model to GPU (if available)
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # needed for early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # set gradient to zero before backpropagation (important!)
            optimizer.zero_grad()

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            # determine gradients through backpropagation
            loss.backward()

            # update model weights based on gradients
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        # disable gradient calculation during validation (save mempry and computations)
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        # average loss over all validation batches
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f},Val Loss: {avg_val_loss:.4f}")

        # early stopping: if validation loss improves, save model state and reset patience counter. if not,
        # reset patience counter save model state.
        # else: increase patience until limit is reached. in this case,
        # stop training and load the best model state.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

    # evaluate model on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            # get predicted morphology class (the one with the highest output score)
            # and compare to true label
            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(batch_labels.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == labels_max).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

    return model

#same as above, but with mixed precision for faster training on compatible hardware (e.g. NVIDIA GPUs with Tensor Cores).
# mixed precision uses float16 float32 for certain operations, which can speed up training and reduce memory usage
# without significantly affecting model performance. the GradScaler is used to prevent underflow during backpropagation
# when using float16.
# drawback: test accuracy is notably lower.
def train_model_mp(ml_data, epochs=10, batch_size=32, learning_rate=0.001, val_split=0.15, test_split=0.15, patience=5, use_mixed_precision=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GradScaler nur bei CUDA und mixed precision aktivieren
    scaler = GradScaler() if device.type == 'cuda' and use_mixed_precision else None
    if scaler is not None:
        print("Mixed Precision Training aktiviert")

    images, labels = ml_data
    num_classes = len(labels[0])

    # convert data to tensors
    dataset = TensorDataset(images, labels)

    # Train/Val/Test Split
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader with Windows-compatible settings
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # use the pre-trained resnet18 model and modify the final layer
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # needed for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Mixed Precision: autocast wrapper um forward pass
            if scaler is not None:
                with autocast():
                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)

                # Skalierter Backward Pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard Training ohne mixed precision
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                if scaler is not None:
                    with autocast():
                        outputs = model(batch_images)
                        loss = criterion(outputs, batch_labels)
                else:
                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                model.load_state_dict(best_model_state)
                break

    # evaluate model on test set
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            if scaler is not None:
                with autocast():
                    outputs = model(batch_images)
                    loss = criterion(outputs, batch_labels)
            else:
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, labels_max = torch.max(batch_labels.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == labels_max).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

    return model


# alternative on windows: data = load_galaxy_data(r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1\training_solutions_rev1.csv', r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\images_training_rev1')
data = load_galaxy_data(r'/mnt/c/Users/olive/Documents/python_codes/a_github_collection/galaxy_morphology_classification/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv', r'/mnt/c/Users/olive/Documents/python_codes/a_github_collection/galaxy_morphology_classification/galaxy-zoo-the-galaxy-challenge/images_training_rev1')
# ml_data = get_galaxy_batch(data, 1000, transform=transforms.Grayscale(num_output_channels=1))

if __name__ == "__main__":

    set_seed(42)

    print("Starting training...", flush=True)
    ml_data = get_galaxy_batch(data, 500)

    start_time = time.time()
    model = train_model(ml_data, epochs=20, batch_size=32)
    end_time = time.time()

    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training completed in {hours}h {minutes}m {seconds}s", flush=True)