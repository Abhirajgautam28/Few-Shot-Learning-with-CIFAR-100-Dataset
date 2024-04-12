import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# Function for plotting
def plot_and_save_metric(metric_values, metric_name, filename):
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6))
    plt.plot(metric_values, label=f'{metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_name}')
    plt.title(f'Model {metric_name} over Epochs')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Function to save images
def save_image_grid(images, labels, predicted, class_labels, save_path):
    num_images = len(images)
    num_rows = (num_images + 2) // 3
    fig, axs = plt.subplots(num_rows, 3, figsize=(12, num_rows * 4))
    axs = axs.flatten() if num_images > 1 else [axs]

    for i, ax in enumerate(axs):
        if i >= num_images:
            break
        img = (images[i].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"Pred: {class_labels[predicted[i]]}\nAct: {class_labels[labels[i]]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Function to create confusion matrix
def save_confusion_matrix(true_labels, predicted_labels, class_labels, save_path):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


# Define CIFAR10Model class with Instance Normalization
class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Define Siamese network for few-shot learning
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = CIFAR10Model()

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)

        return output1, output2


# Contrastive loss for Siamese network
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * 0.5 * euclidean_distance.pow(2) + \
               label * 0.5 * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        return loss.mean()


def main():
    transform_pipeline = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets and data loaders
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_pipeline)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_pipeline)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    # Add a few-shot dataset using the CIFAR-10 dataset
    few_shot_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(range(100))
    )

    class_labels = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize models, loss function, and optimizer
    model = CIFAR10Model()
    siamese_model = SiameseNetwork()
    loss_fn = nn.CrossEntropyLoss()
    contrastive_loss_fn = ContrastiveLoss(margin=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer_siamese = optim.Adam(siamese_model.parameters(), lr=0.001)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    siamese_model.to(device)

    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []

    # Train the model and the Siamese network
    total_epochs = 500

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, unit='batch'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted_classes = outputs.max(1)
            total_samples += labels.size(0)
            correct_preds += predicted_classes.eq(labels).sum().item()

        train_accuracy = 100 * correct_preds / total_samples
        print(
            f'Epoch [{epoch + 1}/{total_epochs}], Training Loss: {train_loss / len(train_loader):.3f}, Training Accuracy: {train_accuracy:.2f}%')

        # Train the Siamese network for few-shot learning
        siamese_model.train()
        siamese_loss = 0.0

        for data1, label1 in tqdm(few_shot_data_loader, unit='batch'):
            data1, label1 = data1.to(device), label1.to(device)
            data2, label2 = data1, label1

            noise = torch.randn_like(data2) * 0.1
            data2 = data2 + noise

            optimizer_siamese.zero_grad()
            output1, output2 = siamese_model(data1, data2)
            label = (label1 != label2).float()
            loss = contrastive_loss_fn(output1, output2, label)
            loss.backward()
            optimizer_siamese.step()

            siamese_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{total_epochs}], Siamese Network Loss: {siamese_loss:.3f}')

        # Evaluate model performance
        model.eval()
        correct_preds = 0
        total_samples = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, unit='batch'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted_classes = outputs.max(1)
                correct_preds += predicted_classes.eq(labels).sum().item()
                total_samples += labels.size(0)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted_classes.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(
            f'Epoch [{epoch + 1}/{total_epochs}], Testing Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}')

        lr_scheduler.step()

    # Save the trained model and Siamese network
    torch.save({'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               'cifar10_model.pth')
    torch.save({'state_dict': siamese_model.state_dict(),
                'optimizer_state_dict': optimizer_siamese.state_dict()},
               'siamese_model.pth')

    # Plot and save metrics
    plot_and_save_metric(accuracy_scores, 'Accuracy', 'accuracy_plot.png')
    plot_and_save_metric(precision_scores, 'Precision', 'precision_plot.png')
    plot_and_save_metric(recall_scores, 'Recall', 'recall_plot.png')
    plot_and_save_metric(f1_scores, 'F1-score', 'f1_score_plot.png')

    # Testing the model and saving predictions as images
    num_images_to_test = 50
    subset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                sampler=torch.utils.data.SubsetRandomSampler(range(num_images_to_test)))

    correct_predictions = 0
    total_samples = 0
    images_to_save = []
    labels_to_save = []
    predicted_to_save = []
    os.makedirs('output_images', exist_ok=True)

    # Iterate through the subset and collect predictions for testing
    for idx, (images, labels) in enumerate(subset_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted_classes = outputs.max(1)

        images_to_save.append(images[0])
        labels_to_save.append(labels[0])
        predicted_to_save.append(predicted_classes[0])

        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += labels.size(0)

        # Save images in grids of 6
        if (idx + 1) % 6 == 0 or idx + 1 == num_images_to_test:
            grid_number = (idx + 1) // 6
            save_image_grid(images_to_save, labels_to_save, predicted_to_save, class_labels,
                            f'output_images/grid_{grid_number}.png')

            images_to_save.clear()
            labels_to_save.clear()
            predicted_to_save.clear()

    accuracy = 100 * correct_predictions / total_samples
    print(f'Accuracy on {num_images_to_test} images: {accuracy:.2f}%')

    # Save confusion matrix
    save_confusion_matrix(true_labels, predicted_labels, class_labels, 'confusion_matrix.png')
    print("Images and confusion matrix saved successfully.")


if __name__ == "__main__":
    main()
