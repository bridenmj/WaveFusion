from Training_Loops import *
from Models import *
from Dataloader import wavelet_dataloader
import time
import os
import copy
import torch
from tqdm.notebook import trange, tqdm

if __name__ == "__main__":

    batch_size = 10
    learning_rate =0.001
    weight_decay = 0.1
    momentum = 0.0005
    num_epochs = 20
    beta = 0.25

    data_dir= "~/1sec_seg_training_class_fif_5000"
    save = True
    save_as = "~/WaveFusion_5000_20ep"
    load_wts = False
    path_wts = "" # "Saved_Models"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loaders_dict = {x: wavelet_dataloader(data_dir = os.path.join(data_dir,x), batch_size = batch_size, shuffle=True) for x in ['train', 'val']}

    Wave_model = Wave_Fusion_Model(beta = beta).to(device)

    lossfun = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(Wave_model.parameters(), lr=0.1 ,weight_decay = weight_decay)

    Wave_model, history = train_model(model = Wave_model,
        dataloaders = data_loaders_dict,
        lossfun = lossfun,optimizer = optim,
        wts_path = path_wts,
        epochs=num_epochs,
        load_wts = load_wts,
        save_as = save_as,
        save = save)
