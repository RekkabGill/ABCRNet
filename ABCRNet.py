import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from multiprocessing import Pool

class ABCRNET(torch.nn.Module):
    '''
    ABCRNet, predicting the compartment membership ratio from DNA sequence
    '''

    def __init__(self):
        super(ABCRNET, self).__init__()
        self.loss = None
        self.optimizer = None
        
        #CNN layers
        self.conv1 = torch.nn.Conv1d(in_channels = 4, out_channels = 64, kernel_size=11, stride=4, padding = 5)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)

        self.conv2 = torch.nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size =21, stride =4, padding = 10)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)

        self.pool1 = torch.nn.AvgPool1d(kernel_size=15625, stride=1, padding=0)

        #Fully Connected layers
        self.fc1 = torch.nn.Linear(128,1)
        torch.nn.init.xavier_uniform_(self.fc1.weight)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool1(x)

        # Resize the conv layer to fit into our fully connected
        # (-1 takes on the size of the previous layer)
        # Then resize it to 
        x = x.view(-1,128)

        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x
