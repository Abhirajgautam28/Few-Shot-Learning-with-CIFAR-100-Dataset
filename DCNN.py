import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


# Function to plot and save metric images
def plot_metrics(metrics_dict, filename):
    sns.set(style='whitegrid')
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics_dict.items():
        plt.plot(values, label=metric_name)

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Performance Metrics')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets and data loaders for training and testing
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=2)

    class_labels = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define the neural network model
    class CIFAR10Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
            self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
            self.pooling_layer = nn.MaxPool2d(2, 2)
            self.conv_dropout = nn.Dropout2d(0.25)
            self.conv_bn1 = nn.BatchNorm2d(128)
            self.conv_bn2 = nn.BatchNorm2d(128)
            self.conv_bn3 = nn.BatchNorm2d(256)
            self.conv_bn4 = nn.BatchNorm2d(256)
            self.fc1 = nn.Linear(256 * 8 * 8, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 10)
            self.fc_dropout = nn.Dropout(0.5)
            self.fc_bn1 = nn.BatchNorm1d(1024)
            self.fc_bn2 = nn.BatchNorm1d(512)

        def conv_stack(self, x):
            x = F.relu(self.conv_bn1(self.conv1(x)))
            x = F.relu(self.conv_bn2(self.conv2(x)))
            x = self.pooling_layer(x)
            x = self.conv_dropout(x)
            x = F.relu(self.conv_bn3(self.conv3(x)))
            x = F.relu(self.conv_bn4(self.conv4(x)))
            x = self.pooling_layer(x)
            x = self.conv_dropout(x)
            return x

        def forward(self, x):
            x = self.conv_stack(x)
            x = x.view(-1, 256 * 8 * 8)
            x = F.relu(self.fc_bn1(self.fc1(x)))
            x = self.fc_dropout(x)
            x = F.relu(self.fc_bn2(self.fc2(x)))
            x = self.fc_dropout(x)
            x = self.fc3(x)
            return x

    # Initialize model, loss function, and optimizer
    model = CIFAR10Network()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    learning_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                              verbose=True)

    # Check for device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and testing loops
    total_epochs = 500
    train_losses, test_losses = [], []
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in tqdm(train_loader, unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_classes = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted_classes.eq(labels).sum().item()

        train_accuracy = correct_train / total_train * 100
        print(
            f'Epoch [{epoch + 1}/{total_epochs}], Training Loss: {train_loss / len(train_loader):.3f}, Training Accuracy: {train_accuracy:.2f}%')

        # Testing loop
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, unit='batch'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                _, predicted_classes = outputs.max(1)
                correct_test += predicted_classes.eq(labels).sum().item()
                total_test += labels.size(0)

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted_classes.cpu().numpy())

        test_accuracy = correct_test / total_test * 100

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        # Store metrics
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))

        learning_scheduler.step(test_loss)
        print(
            f"Epoch: {epoch + 1}/{total_epochs}, Training Loss: {train_loss / len(train_loader):.3f}, Testing Loss: {test_loss / len(test_loader):.3f}")

    # Save the trained model
    saved_model_state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(saved_model_state, 'CIFAR10_trained_model.pth')

    # Load the trained model
    saved_model_state = torch.load('CIFAR10_trained_model.pth')
    model = CIFAR10Network()
    model.load_state_dict(saved_model_state['state_dict'])
    optimizer.load_state_dict(saved_model_state['optimizer'])

    # Plot and save metrics
    metrics_dict = {
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores
    }
    plot_metrics(metrics_dict, 'performance_metrics.png')

    # Testing and saving predictions as images
    num_images_to_test = 50
    subset_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                sampler=torch.utils.data.SubsetRandomSampler(range(num_images_to_test)))

    # Calculate accuracy
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(subset_loader, unit='batch'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_classes = outputs.max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted_classes.eq(labels).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f'Accuracy on {num_images_to_test} images: {accuracy:.2f}%')

    # Save predictions as images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Create a directory for saving images
    os.makedirs('output_images', exist_ok=True)

    for i, (image, label) in enumerate(subset_loader):
        if i >= num_images_to_test:
            break
        with torch.no_grad():
            output = model(image.to(device))
            _, predicted_classes = torch.max(output, 1)

        # Save each image with the filename indicating predicted and actual class
        filename = f'output_images/{i}_pred_{class_names[predicted_classes[0]]}_act_{class_names[label[0]]}.png'
        image = (image[0] * 0.5 + 0.5).permute(1, 2, 0).numpy()  # Rescale and convert tensor to numpy array
        plt.imsave(filename, image)

    print("Images saved successfully.")


if __name__ == "__main__":
    main()
