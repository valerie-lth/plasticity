import random
import pathlib
import argparse
from tqdm import tqdm
from dateutil import tz
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush


class LeNet5(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        if activation == 'relu':
            activation_fn = nn.ReLu()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.Identity()

        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            activation_fn,
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            activation_fn,
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            activation_fn,
            nn.Linear(in_features=120, out_features=84),
            activation_fn,
            nn.Linear(in_features=84, out_features=10),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.fc5 = nn.Linear(hidden_units, output_dim)

    # both relu and leaky rely suffer from loss of plasticity
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # Linear activation for regression
        return x
    
def shuffle_subset_label(subset, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    labels = [subset[i][1] for i in range(len(subset))]
    shuffled_labels = torch.randperm(len(labels))  # Shuffle the labels, note this returns label indices
    subset_with_shuffled_labels = [(subset[i][0], labels[shuffled_labels[i]]) for i in range(len(subset))]  # Combine images with shuffled labels
    return subset_with_shuffled_labels


def uniform_random_mnist(dataset, num_sample_per_class, classes, seed=0):
    '''
    Samples even number of samples for each class
    '''
    random.seed(seed)
    sampled_indices = []
    
    for class_label in classes:
        class_indices = torch.where(dataset.targets == class_label)[0]

        sampled_indices.extend(random.sample(class_indices.tolist(), num_sample_per_class))
    
    subset = Subset(dataset, sampled_indices)
    return subset
    
    
def train_task(optimizer, criterion, writer, epochs=200, batch_size=256):
    # randomly shuffle MNIST labels for every task
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
    ])
    train_dataset = datasets.MNIST(root='./data', 
                            train=False, 
                            transform=transform,
                            target_transform=lambda y: torch.randint(0, 10, (1,)).item(),
                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    total_batch = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # log at the very last batch
            if batch_idx % total_batch == 0:
                avg_loss = running_loss / batch_size
                
                writer.add_scalar('train/loss', avg_loss, epoch)
                # print('[Epoch %d, loss: %.6f' %
                #     (epoch + 1, running_loss / batch_size))
                running_loss = 0.0

if __name__=="__main__":

    parser = argparse.ArgumentParser('random mnist')

    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-per-class', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--save-every-task', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default="log_random_lenet/")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--activation", type=str, default='tanh')

    
    args = parser.parse_args()
    logger = Logger("tasks.log")
    sys.stdout = logger
    sys.stderr = logger

    save_root_dir = "models/"
    tzone = tz.gettz('America/Edmonton')
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    if args.save:
        save_name = "random_mnist".format(args.num_per_class, args.num_tasks, args.lr)
        save_dir = pathlib.PosixPath(save_root_dir, save_name + '_' + timestamp)
        save_dir.mkdir()

    # set global seed
    seed = 0
    torch.random.manual_seed(seed)
    random.seed(seed)

    model = LeNet5(args.activation).to(device)
    
    # default Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # mnist does not have a uniform distribution
    classes = list(range(args.num_classes))
    num_samples_per_class = args.num_per_class

    wandb.init(
        # set the wandb project where this run will be logged
        project="plasticity-project",

        # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "Lenet5",
            "epochs": args.num_epochs,
            "num_tasks" : args.num_tasks,
        }
    )


    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        # transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize pixel values to range [0, 1]
    ])
    
    mnist_train = datasets.MNIST(root='./data', 
        train=True, 
        transform=transform,
        
        download=True)
    
    # mnist_test = datasets.MNIST(root='./data', 
    #     train=False, 
    #     transform=transform,
    #     # target_transform=lambda y: torch.randint(0, 10, (1,)).item(),
    #     download=True)
    

    train_uniform_mnist_subset = uniform_random_mnist(mnist_train, num_samples_per_class, classes)
    # test_uniform_mnist_subset = uniform_random_mnist(mnist_test, num_samples_per_class_test, classes)

    num_tasks = args.num_tasks
    epochs = args.num_epochs
    batch_size = args.batch_size

    global_epoch = 0
    
    for task in range(num_tasks):
        print("Starting task ", task)

        # shuffle label from every task
        random_label_train_subset = shuffle_subset_label(train_uniform_mnist_subset, seed)
        seed += 1
        # random_label_test_subset = shuffle_subset_label(test_uniform_mnist_subset)
        train_loader = DataLoader(random_label_train_subset, batch_size, shuffle=True)
        # test_loader = DataLoader(random_label_test_subset, batch_size, shuffle=True)
        total_batch = len(train_loader)
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, 1, 28, 28).to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
        
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (predicted == labels).sum().item()
                # the running loss is the batch average loss
                epoch_loss += loss.item()
            
                # log at the very last batch
                if batch_idx == total_batch - 1:
                    # record the average batch loss and accuracy
                    wandb.log({"train_loss": epoch_loss / total_batch, "global_epoch": global_epoch})
                    wandb.log({"train_acc": correct / len(train_uniform_mnist_subset), "global_epoch": global_epoch})
            global_epoch += 1
        if task % args.save_every_task == 0:
            if args.save:
                save_file = "task{}.pt".format(task)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'global_epoch': global_epoch,
                    'learning_rate': args.lr
                }
                torch.save(checkpoint, pathlib.PosixPath(save_dir, save_file))

        # # test phase
            # model.eval()
            # test_loss_sum = 0
            # num_test_batch = len(test_loader)
            # with torch.no_grad():
            #     n_correct = 0
            #     n_samples = 0
            #     for images, labels in test_loader:
            #         # conv2d expects to see shape: [batch_size, channels, height, width]
            #         images = images.reshape(-1, 1, 28, 28).to(device)
            #         # images = images.reshape(-1, 28*28).to(device)
            #         labels = labels.to(device)
            #         outputs = model(images)
            #         loss = criterion(outputs, labels)
            #         test_loss_sum += loss.item()
            #         # max returns (value ,index)
            #         _, predicted = torch.max(outputs.data, 1)
            #         n_samples += labels.size(0)
            #         n_correct += (predicted == labels).sum().item() 
            #     writer.add_scalar('epoch_test/loss', test_loss_sum / num_test_batch, global_epoch)
            #     writer.add_scalar('epoch_test/acc', n_correct / n_samples, global_epoch)
    
                    