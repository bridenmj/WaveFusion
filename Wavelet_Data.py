import numpy as np
import csv
import os
import pandas as pd
import copy
from scipy import signal
import torch
from scipy import signal
import pywt
import mne
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoader


class HiddenPrints:
    """
    Helper class to suppress mne print statements
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        

        
        
class Wavelet_Dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = self.make_dataset(root_dir, ".fif")
        self.transform = transform

    def make_dataset(self, dir , extensions):
        """creates a list of paths to data in root data directory

        Args:
            dir: path to data directory of train/val datafiles i.e. ~/LEMON/train
            class_to_index

        Returns:
            list: images list of pats to data
        """

        images = []

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if self.has_file_allowed_extension(fname, ".fif"):
                    path = os.path.join(root, fname)
                    cls = -1
                    if fname.split('_')[2] == "mild":
                        cls = 0
                    elif fname.split('_')[2] == "moderate":
                        cls = 1
                    elif fname.split('_')[2] == "severe":
                        cls = 2

                    item = (path,cls)
                    #print(item)
                    images.append(item)

        return images

    def Wavelet(self,path:str):
        """
        create data cubes using data from .fif file at path. of the form 
        (Lead)x(scale)x(time).

        Args:
            path to .fif file (should be passed by Wavelet and dataset)

        Returns: wave_mag 3D numpy array containing time frequency transform data
                 of shape (Lead)x(scale)x(time).
        """
        channels = 61
        scale = 32 #scale param for morle wavelet
        length = 250

        with HiddenPrints():
            raw = mne.io.read_raw_fif(path)
        raw = raw.get_data(picks=raw.ch_names, start=0)

        data = pd.DataFrame(data=raw)

        y_voltage = np.empty([channels, length])
        for i in range(channels):
            y_voltage[i] = data.iloc[i]


        # Generate a 3d array of channelXfreqXtime
        waves_mag = np.empty([61, scale, 250 ], dtype=float) 

        # compute and store complex morlet transform for each lead
        # THIS VERSION PUTS Lead First
        for i in range(channels):
            coef, freqs=pywt.cwt(y_voltage[i],np.arange(1,scale+1),'cmor0.4-1.0',sampling_period=1)
            waves_mag[i,:,:] = copy.deepcopy(abs(coef))

        return waves_mag
        
    def has_file_allowed_extension(self, filename, extensions):
        """Generic file extension checker. does

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in extensions)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (Wavelet(sample(index)), target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.Wavelet(path)
        sample = np.array(sample).astype(np.float64)
        sample = torch.FloatTensor(sample)
        
        if self.transform:
            sample = self.transform(sample)

        return sample, target
    
    def __len__(self):

        return len(self.samples)
    
