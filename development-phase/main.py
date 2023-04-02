### Import packages
import os
import argparse

# Pytorch
import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader


from sklearn.metrics import confusion_matrix
import numpy as np


# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from models import resnet_model, custom_model


### Arguments
parser = argparse.ArgumentParser(description='Train ResNet on CIFAR10 dataset')
parser.add_argument('--model', default="resnet18-v0", type=str, help='Model used for training')
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for DataLoader')
args = parser.parse_args()

### Data
data_dir = './data'
classes = os.listdir(data_dir + "/train")

# Transform Image to Tensor
dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())

###  Training and Validation Dataset

random_seed = 26
torch.manual_seed(random_seed)

# Split data into Train and Validation dataset
val_size = int(0.1*(len(dataset)))            # set validation dataset size is 10% of train dataset size
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Set batch size into training & validation data loader
batch_size = args.batch_size

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

### Test Dataset

test_dl = DataLoader(test_dataset, batch_size*2)

### Model Dictionary

model_dict = {
    "custom-v1":custom_model.Cifar10CnnModel(),
    "resnet18-v0":resnet_model.PreTrainedResNet()
}

model=model_dict[args.model]

### Training Model

@torch.no_grad()
def evaluate(model=model, val_loader=val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs=args.epochs, lr=args.lr, model=model, train_loader=train_dl, val_loader=val_dl, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        train_acc = []
        for batch in train_loader:
            loss = model.training_step(batch)[0]
            acc = model.training_step(batch)[1]
            train_losses.append(loss)
            train_acc.append(acc)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item() if train_losses else float('nan')
        result['train_acc'] = torch.stack(train_acc).mean().item() if train_acc else float('nan')
        model.epoch_end(epoch, result)
        history.append(result)
    return history


### Test
def evaluate_test_dataset(model=model, test_loader=test_dl):
    result = evaluate(model, test_loader)

    return print("Test Loss: {:.4f} ;\nTest Accuracy: {:.4f}".format(result['val_loss'], result['val_acc']))

def calc_test_accuracy(model=model, test_loader=test_dl):
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
          for batch in test_loader:
              images, labels = batch
              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              true_labels += labels.tolist()
              pred_labels += predicted.tolist()
    cm = confusion_matrix(true_labels, pred_labels)
    return cm

### Chart Evaluation
def plot_evaluation(history, classes=classes):
    fig = plt.figure(figsize=(16, 6))

    # Plot the training and validation losses
    ax1 = fig.add_subplot(2, 2, 1)
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    ax1.plot(train_losses, '-bx')
    ax1.plot(val_losses, '-rx')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend(['Training Loss', 'Validation Loss'])
    ax1.set_title('Loss vs. No. of epochs')

    # Plot the training and validation accuracies
    ax2 = fig.add_subplot(2, 2, 2)
    train_acc = [x.get('train_acc') for x in history]
    val_acc = [x['val_acc'] for x in history]                
    ax2.plot(train_acc, '-bx')
    ax2.plot(val_acc, '-rx')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend(['Training Accuracy', 'Validation Accuracy'])
    ax2.set_title('Accuracy vs. No. of epochs')

    plt.subplots_adjust(hspace=0.5)
    
    # Add the confusion matrix
    ax3 = fig.add_subplot(2, 1, 2)
    sns.heatmap(calc_test_accuracy(), annot=True, cmap=plt.cm.Blues, fmt='g', cbar=False, ax=ax3)
    tick_marks = np.arange(len(classes))
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(classes, rotation=45)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(classes, rotation=0)
    ax3.set_xlabel('Predicted labels')
    ax3.set_ylabel('True labels')
    ax3.set_title('Confusion Matrix')

    plt.show()

### Save model
model_scripted = torch.jit.script(model)
model_scripted.save('../deployement-phase/app/model1.pt')

if __name__ == '__main__' :
    history = fit()
    evaluate_test_dataset()
    plot_evaluation(history)