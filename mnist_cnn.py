import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def show_sample_images():
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, (image, label) in enumerate(train_dataset):
        if i >= 10:
            break
        row = i // 5
        col = i % 5
        axes[row, col].imshow(image.squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {label}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    return running_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy, all_preds, all_targets

def train_with_config(optimizer_name, loss_name, epochs=5):
    model = CNN().to(device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'nll':
        criterion = nn.NLLLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc, all_preds, all_targets = test(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

    return model, history, all_preds, all_targets

def plot_training_history(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for config, result in results.items():
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0, 0].plot(epochs, history['train_loss'], label=f'{config} (train)')
        axes[0, 0].plot(epochs, history['test_loss'], label=f'{config} (test)')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, history['train_acc'], label=f'{config} (train)')
        axes[0, 1].plot(epochs, history['test_acc'], label=f'{config} (test)')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(predictions, targets, title):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    print("Displaying sample images...")
    show_sample_images()

    configs = [
        ('adam', 'cross_entropy'),
        ('sgd', 'cross_entropy'),
        ('rmsprop', 'cross_entropy')
    ]

    results = {}
    for opt_name, loss_name in configs:
        print(f'\nTraining with {opt_name} optimizer and {loss_name} loss function')
        model, history, preds, targets = train_with_config(opt_name, loss_name)
        results[f'{opt_name}_{loss_name}'] = {
            'model': model,
            'history': history,
            'predictions': preds,
            'targets': targets
        }

    print("\nPlotting training history...")
    plot_training_history(results)

    print("\nPlotting confusion matrices...")
    for config, result in results.items():
        plot_confusion_matrix(
            result['predictions'],
            result['targets'],
            f'Confusion Matrix - {config}'
        )

if __name__ == '__main__':
    main()