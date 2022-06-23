from Training_Loops import *
from Models import *
from Dataloader import wavelet_dataloader
import time
import os
import copy
import torch
from tqdm.notebook import trange, tqdm
from torch import nn

if __name__ == "__main__":

batch_size = 500
learning_rate =0.001
weight_decay = 7.5e-4
momentum = 0.0005
num_epochs = 35

#save model?
save = False
#ex: "save_as = <fileName>.pt" or "save_as = <fileName>.pth"
save_as = ""

#load wts from torch.nn.Module.load_state_dict for "model"?
load_wts = False

#path torch.nn.Module.load_state_dict for "model"
path_wts = "" # "Saved_Models"

#set to train w/ GPU if available else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = xm.xla_device()
print(device)

    wavelet_datasets = {x: Wavelet_Dataset(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    data_loaders_dict = {x: wavelet_dataloader(dataset=wavelet_datasets[x], batch_size = batch_size, shuffle=True) for x in ['train', 'val']}

    Wave_model = Wave_Fusion_Model( device = device).to(device)
    
    optim = torch.optim.Adam(Wave_model.parameters(), lr=learning_rate , weight_decay=weight_decay)
    lossfun = nn.CrossEntropyLoss()

    model, history = train_model(model = Wave_model, dataloaders = data_loaders_dict, loss = lossfun, save = save, save_as = save_as, optimizer = optim, epochs=num_epochs,load_wts = load_wts, wts = path_wts)
