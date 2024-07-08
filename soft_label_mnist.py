import random
import pathlib
import argparse
from tqdm import tqdm
from dateutil import tz
import datetime
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
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
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    
def smooth_labels(subset, num_classes=10, smoothing=0.1):
    new_subset = []
    for i in range(len(subset)):
        image, orig_label = subset[i]
        # first filled with smoothing / num_classes
        soft_label = np.full((1, num_classes),  smoothing/(num_classes-1))
        # Assign (1 - smoothing) to the true label index
        soft_label[0][orig_label] = 1-smoothing
        new_subset.append((image, soft_label[0]))
       
    return new_subset


def shuffle_subset_label(subset, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    labels = [subset[i][1] for i in range(len(subset))]
    shuffled_labels = torch.randperm(len(labels))  # Shuffle the labels, note this returns label indices
    subset_with_shuffled_labels = [(subset[i][0], labels[shuffled_labels[i]]) for i in range(len(subset))]  # Combine images with shuffled labels
    return subset_with_shuffled_labels
    

def shuffle_class_label(train_subset, test_subset, classes, seed):
    '''
    Map label of one class entirely to another class
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # shuffle the order of classes (without replacement)
    random_order_classes = random.sample(classes, len(classes))
    random_label_map = mapping = {k: v for k, v in zip(classes, random_order_classes)}
    # change the class label of original mnist to random class
    new_train_subset = []
    for i in range(len(train_subset)):
        image, label = train_subset[i]
        if label in mapping:
            new_train_subset.append((train_subset[i][0], random_label_map[train_subset[i][1]]))
    new_test_subset = []
    for i in range(len(test_subset)):
        image, label = test_subset[i]
        if label in mapping:
            new_test_subset.append((test_subset[i][0], random_label_map[test_subset[i][1]]))
    return new_train_subset, new_test_subset

def uniform_random_mnist(dataset, num_sample_per_class, classes, seed=0):
    '''
    Samples even number of samples for each class
    '''
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampled_indices = []
    
    for class_label in classes:
        class_indices = torch.where(dataset.targets == class_label)[0]

        sampled_indices.extend(random.sample(class_indices.tolist(), num_sample_per_class))
    
    subset = Subset(dataset, sampled_indices)
    return subset
    # sampler = SubsetRandomSampler(sampled_indices)

def eval(test_loader, global_epoch):
    model.eval()
    test_loss_sum = 0
    num_test_batch = len(test_loader)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            # conv2d expects to see shape: [batch_size, channels, height, width]
            images = images.reshape(-1, 1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_sum += loss.item()
            n_samples += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = torch.max(labels, 1)
            n_correct += (predicted == true_labels.indices).sum().item()
            
        wandb.log({"test_loss": test_loss_sum / num_test_batch, "global_epoch": global_epoch})
        wandb.log({"test_acc": n_correct / n_samples, "global_epoch": global_epoch})

def stair_task_change(train_uniform_mnist_subset, args):
    global_epoch = 0
    seed=0
    random_train = shuffle_subset_label(train_uniform_mnist_subset, seed)
    seed += 1

    # increase the smooth factor until uniform probabilities
    for smooth in np.arange(0, 1, args.smooth_inc):
        for task in range(num_tasks):
            # print("Starting task ", task)
            random_class_train_soft_subset = smooth_labels(random_train, args.smooth_type, num_classes=args.num_classes, smoothing=smooth)
            train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
        
            total_train_batch = len(train_loader)
            for epoch in range(epochs):
                epoch_loss = 0
                correct = 0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, 1, 28, 28).to(device)

                    # this label should be a vector instead of a integer
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    true_labels = torch.max(labels, 1)
                
                    loss = criterion(outputs, labels)   
                    loss.backward()
                    optimizer.step()

                    correct += (predicted == true_labels.indices).sum().item()
                    # the running loss is the batch average loss
                    epoch_loss += loss.item()
                
                    # log at the very last batch
                    if batch_idx == total_train_batch - 1:
                        # record the average batch loss and accuracy
                        wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                        wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                # eval(test_loader, global_epoch)
                global_epoch += 1

def continuous_task_change(train_uniform_mnist_subset, args):
    '''
        Train on 0 smoothing for certain number of epochs
        Then use k * num_epochs as a transition stage where the smooth factor increase 0.9/k * num_epochs after each epoch
    '''
    k = args.k
    global_epoch = 0
    seed = 0
    random_train = shuffle_subset_label(train_uniform_mnist_subset, seed)
    seed += 1
    
    for task in range(num_tasks):
        print("Starting task ", task)
        
        smooth_factor = 0
        random_class_train_soft_subset = smooth_labels(random_train, args.smooth_type, num_classes=args.num_classes, smoothing=smooth_factor)
        train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
    
        total_train_batch = len(train_loader)
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, 1, 28, 28).to(device)

                # this label should be a vector instead of a integer
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                true_labels = torch.max(labels, 1)
            
                loss = criterion(outputs, labels)   
                loss.backward()
                optimizer.step()

                correct += (predicted == true_labels.indices).sum().item()
                # the running loss is the batch average loss
                epoch_loss += loss.item()
            
                # log at the very last batch
                if batch_idx == total_train_batch - 1:
                    # record the average batch loss and accuracy
                    # wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                    # wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                    print("train_loss/global_epoch ", epoch_loss / total_train_batch)
                    print("train_acc/global_epoch ", correct / len(random_train))
            global_epoch += 1
        
        # save here for each complete task
        if args.save:
            save_file = "task{}.pt".format(task)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_epoch': global_epoch,
                'learning_rate': args.lr,
                'k': args.k,
                'seed': seed
            }
            torch.save(checkpoint, pathlib.PosixPath(save_dir, save_file))
        # transition stage to uniform distribution
        for epoch in range(int(k/2 * epochs)):
            smooth_factor += 0.9/(k/2*epochs)
            
            random_class_train_soft_subset = smooth_labels(random_train, args.smooth_type, num_classes=args.num_classes, smoothing=smooth_factor)
            train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
        
            total_train_batch = len(train_loader)
            epoch_loss = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, 1, 28, 28).to(device)

                # this label should be a vector instead of a integer
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                true_labels = torch.max(labels, 1)
            
                loss = criterion(outputs, labels)   
                loss.backward()
                optimizer.step()

                # wandb.log({"batch/train_loss": loss.item(), "global_step": global_step})
                # global_step += 1

                correct += (predicted == true_labels.indices).sum().item()
                # the running loss is the batch average loss
                epoch_loss += loss.item()
            
                # log at the very last batch
                if batch_idx == total_train_batch - 1:
                    # record the average batch loss and accuracy
                    # wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                    # wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                    print("train_loss/global_epoch ", epoch_loss / total_train_batch)
                    print("train_acc/global_epoch ", correct / len(random_train))
            # eval(test_loader, global_epoch)
            global_epoch += 1

        # task change at uniform distribution
        random_train = shuffle_subset_label(train_uniform_mnist_subset, seed)
        seed += 1

        # transition stage to 0 smoothing (gradually decrease smooth_factor from 0.9 to 0)
        for epoch in range(int(k/2 * epochs)):
            smooth_factor -= 0.9/(k/2*epochs)
            random_class_train_soft_subset = smooth_labels(random_train, args.smooth_type, num_classes=args.num_classes, smoothing=smooth_factor)
            train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
        
            total_train_batch = len(train_loader)
            epoch_loss = 0
            correct = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.reshape(-1, 1, 28, 28).to(device)

                # this label should be a vector instead of a integer
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                true_labels = torch.max(labels, 1)
            
                loss = criterion(outputs, labels)   
                loss.backward()
                optimizer.step()

                correct += (predicted == true_labels.indices).sum().item()
                # the running loss is the batch average loss
                epoch_loss += loss.item()
            
                # log at the very last batch
                if batch_idx == total_train_batch - 1:
                    # record the average batch loss and accuracy
                    # wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                    # wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                    print("train_loss/global_epoch ", epoch_loss / total_train_batch)
                    print("train_acc/global_epoch ", correct / len(random_train))

            global_epoch += 1

def gradual_shuffle_task(train_uniform_mnist_subset, args):
    global_epoch = 0
    seed = 0
    random_train = shuffle_subset_label(train_uniform_mnist_subset, seed)
    seed += 1
    
    for task in range(num_tasks):
        print("Starting task ", task)
        # increase the smooth factor until uniform probabilities
        for smooth in np.arange(0, 1, args.smooth_inc):
            random_class_train_soft_subset = smooth_labels(random_train, num_classes=args.num_classes, smoothing=smooth)
            train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
            
            total_train_batch = len(train_loader)
            for epoch in range(epochs):
                epoch_loss = 0
                correct = 0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, 1, 28, 28).to(device)

                    # this label should be a vector instead of a integer
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    true_labels = torch.max(labels, 1)
                
                    loss = criterion(outputs, labels)   
                    loss.backward()
                    optimizer.step()

                    correct += (predicted == true_labels.indices).sum().item()
                    # the running loss is the batch average loss
                    epoch_loss += loss.item()
                
                    # log at the very last batch
                    if batch_idx == total_train_batch - 1:
                        # record the average batch loss and accuracy
                        wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                        wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                global_epoch += 1

        # shuffle the labels randomly 
        random_train = shuffle_subset_label(random_train, seed)
        seed += 1
        
        for smooth in np.arange(1-args.smooth_inc, -args.smooth_inc, -args.smooth_inc):
            random_class_train_soft_subset = smooth_labels(random_train, num_classes=args.num_classes, smoothing=smooth)
            train_loader = DataLoader(random_class_train_soft_subset, args.batch_size, shuffle=True)
        
            total_train_batch = len(train_loader)
            for epoch in range(epochs):
                epoch_loss = 0
                correct = 0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.reshape(-1, 1, 28, 28).to(device)

                    # this label should be a vector instead of a integer
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    true_labels = torch.max(labels, 1)
                
                    loss = criterion(outputs, labels)   
                    loss.backward()
                    optimizer.step()

                    correct += (predicted == true_labels.indices).sum().item()
                    # the running loss is the batch average loss
                    epoch_loss += loss.item()
                
                    # log at the very last batch
                    if batch_idx == total_train_batch - 1:
                        # record the average batch loss and accuracy
                        wandb.log({"train_loss": epoch_loss / total_train_batch, "global_epoch": global_epoch})
                        wandb.log({"train_acc": correct / len(random_train), "global_epoch": global_epoch})
                global_epoch += 1
        if args.save:
            save_file = "task{}.pt".format(task)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_epoch': global_epoch,
                'learning_rate': args.lr,
                'seed': seed
            }
            torch.save(checkpoint, pathlib.PosixPath(save_dir, save_file))

if __name__=="__main__":

    parser = argparse.ArgumentParser('soft label mnist')

    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-per-class', type=int, default=5120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument("--smooth-init", type=float, default=0.1)
    parser.add_argument("--smooth-inc", type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument("--save", action='store_true')

    args = parser.parse_args()
    
    save_root_dir = "models/"
    tzone = tz.gettz('America/Edmonton')
    timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')
    save_name = "soft_label_anneal"
    save_dir = pathlib.PosixPath(save_root_dir, save_name + '_' + timestamp)
    if args.save:
        save_dir.mkdir()

    logger = Logger(pathlib.PosixPath(save_dir, 'tasks.log'))
    sys.stdout = logger
    sys.stderr = logger

    print('> Command:', ' '.join(sys.argv))
    print()

    # set global seed
    seed = 0
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = LeNet5().to(device)
    
    # default Adam
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    classes = list(range(args.num_classes))
    num_samples_per_class = args.num_per_class


    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize pixel values to range [0, 1]
    ])
    
    mnist_train = datasets.MNIST(root='./data', 
        train=True, 
        transform=transform,
        download=True)

    train_uniform_mnist_subset = uniform_random_mnist(mnist_train, num_samples_per_class, classes)
    
    num_tasks = args.num_tasks
    epochs = args.num_epochs
    batch_size = args.batch_size


    if args.continuous:
        continuous_task_change(train_uniform_mnist_subset, args)
    else:
        gradual_shuffle_task(train_uniform_mnist_subset, args)


        
    
                    