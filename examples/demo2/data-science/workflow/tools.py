import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

from torch.utils.data import DataLoader, Dataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def get_model_params():
    params = {
      'test_batch_size': 1000,
    }

    return params

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Consider x and y

def get_dataloader(x, y, batch_size, transform, kwargs):
    
    class CustomDataset(Dataset):
        
        def __init__(self, x, y, transform=None):

            self.length = x.shape[0]
            self.x_data = x
            # self.x_data = torch.from_numpy(x)
            self.y_data = y
            # self.y_data = torch.from_numpy(y)
            self.transform = transform

        def __getitem__(self, index):
            x_data = self.x_data[index]

            if self.transform:
                x_data = self.transform(x_data)

            return (x_data, self.y_data[index])

        def __len__(self):
            return self.length

    train_dataset = CustomDataset(x, y, transform)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, **kwargs)
    
    return train_loader

def get_dataloader_x(x, batch_size, transform, kwargs):
    
    class CustomDataset(Dataset):
        
        def __init__(self, x, transform=None):

            self.length = x.shape[0]
            self.x_data = x
            # self.x_data = torch.from_numpy(x)
            # self.y_data = y
            # self.y_data = torch.from_numpy(y)
            self.transform = transform

        def __getitem__(self, index):
            x_data = self.x_data[index]

            if self.transform:
                x_data = self.transform(x_data)

            # return (x_data, self.y_data[index])
            return x_data

        def __len__(self):
            return self.length

    train_dataset = CustomDataset(x, transform)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, **kwargs)
    
    return train_loader

def predict_model(model, x):
    params = get_model_params()

    model.eval()
    device = get_device()
    
    kwargs = {'num_workers': 1, 'pin_memory': True}

    transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])

    test_loader = get_dataloader_x(x, params['test_batch_size'],
                                transform, kwargs)
  
    preds = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            preds.extend(pred.cpu().detach().numpy().flatten())

    return preds



