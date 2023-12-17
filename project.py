from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from math import sin, cos
from logging import getLogger, basicConfig
import random, sys

device = torch.device("cuda:0")

def get_partial_data(filename: Path, num_train, num_test) -> tuple[torch.tensor,torch.tensor]:
    train = []
    test = []
    for i in filename.iterdir():
        with open(i) as file:
            strs = file.readlines()
            if num_train is None:
                data = strs
            else:
                data = random.choices(strs,k=num_train+num_test)
        data = [tuple(map(float, i.split())) for i in data]
        
        train += data[:num_train]
        test += data[num_train:]
    return (torch.tensor(train,device=device),torch.tensor(test,device=device))

def serialize(a: dict) -> tuple:
    for i in a:
        yield (i['CentrePosition']['X'],i['CentrePosition']['Y'],i['CentrePosition']['Z'],i['Angle'],i['Velocity'],i['PressureValue'])


def unpack_data(data: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    return (torch.cat( ( data[:3],torch.tensor((cos(data[3]),sin(data[3])),device=device), data[-2:-1] ),-1), data[-1:])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 100)   # X Y Z COS SIN VEL
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

logger = getLogger(__name__)

basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level='INFO', datefmt="%H:%M:%S")
