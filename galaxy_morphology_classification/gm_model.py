import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split



from gm_datapipeline import load_galaxy_data, get_galaxy_batch


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

    # DataLoader with Windows-compatible settings. may be better to install wsl2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
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

data = load_galaxy_data(r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1\training_solutions_rev1.csv', r'C:\Users\olive\Documents\python_codes\a_github_collection\galaxy_morphology_classification\galaxy-zoo-the-galaxy-challenge\images_training_rev1')
# ml_data = get_galaxy_batch(data, 1000, transform=transforms.Grayscale(num_output_channels=1))

if __name__ == "__main__":
    ml_data = get_galaxy_batch(data, 5000)
    model = train_model(ml_data, epochs=50, batch_size=256)