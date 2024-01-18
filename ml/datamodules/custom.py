from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import torch

class DataHandler(LightningDataModule):
    """
      Data module for custom dataset.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
        distribution = None,
        train_val_split: Tuple[float, float] = [0.9,0.1],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()
        self.distribution = distribution
        print("data_path",data_path)
        if data_path is not None:
            self.dataset = TrajData(data_path=data_path)
        elif distribution is not None:
            self.dataset = LiveSimulation(distribution=distribution, batch_size=batch_size)
            self.dataset_test = LiveSimulation(distribution=distribution,batch_size=batch_size)
        if test_data_path is not None:
            self.dataset_test = TrajData(data_path=test_data_path)
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        pass
        
    def setup(self, stage: Optional[str] = None):
        """
          Split data and set
        """

        self.data_train, self.data_val,  = random_split(
            self.dataset, self.train_val_split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

class TrajData(Dataset):   
    def __init__(self, data_path=None, encoder=None, flattening=False,**kwargs):
        self.data_path = data_path
        self.encoder = encoder
        self.flattening = flattening
        if data_path is not None:
            self.traj = self.load_traj(data_path)
    def __len__(self):
        return len(self.traj)
    
    def __getitem__(self, idx): 
        data = self.traj[idx]
        if self.encoder is not None:
            data = self.encoder(torch.tensor(data))
        return data

    def update_data(self,file,append=False):
        traj = self.load_traj(file)
        if append:
            self.traj = torch.cat((self.traj,traj),axis=0)
        else:
            self.traj = traj

    def load_traj(self,data_path):
        ext = os.path.splitext(data_path)[-1].lower()
        if ext == ".pt":
            traj = torch.tensor(torch.load(data_path)).float()
        elif ext == ".npy":
            traj = np.load(data_path,mmap_mode='c').astype('float32')
        else:
            raise NotImplementedError
        if self.flattening:
            traj = traj.reshape(len(traj),-1)
        return traj

class SplitTrajData(TrajData):
    def __init__(self, context_dim=0,random_split=False, data_path=None, encoder=None, flattening=False,**kwargs):
        super().__init__(data_path=data_path, encoder=encoder, flattening=flattening,**kwargs)
        self.context_dim = context_dim
        self.random_split = random_split

    def __getitem__(self, idx):
        data = self.traj[idx]
        if self.encoder is not None:
            data = self.encoder(torch.tensor(data))
        if self.context_dim > 0:
            if self.random_split:
                rand_idx = torch.randperm(data.shape[1])
                context = data[:,rand_idx[:self.context_dim]]
                data = data[:,rand_idx[self.context_dim:]]
            else:
                context = data[:self.context_dim]
                data = data[self.context_dim:]
            return data, context
        else:
            return data

class LiveSimulation(Dataset):
    def __init__(self, distribution, batch_size=100,
                  nbatches_per_epoch=500, flattening=False,**kwargs):
        self.flattening = flattening
        self.distribution = distribution
        self.nbatches_per_epoch = nbatches_per_epoch
        self.batch_size = batch_size
        
    def __len__(self):
        return self.nbatches_per_epoch*self.batch_size
    
    def __getitem__(self, idx):    
        data = self.distribution.sample(1).squeeze(0)
        data = torch.tensor(data)
        if self.flattening:
            data = data.reshape(self.batch_size,-1)
        return data


if __name__=="__main__":
    dataset = TrajData(data_path="/home/sherryli/xsli/GenerativeModel/data/lj/liquid_3d/nonperiodic/lj_5_3d_2.0box_T1.00/pos_train.npy")
    data_handler = DataHandler(dataset)
    