from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_sevir import get_sevir_loader
from omegaconf import OmegaConf
from .sevir_torch_wrap import get_sevir_datamodule
import torch
import torch.nn.functional as F
from normalize import zscore_normalizations
import numpy as np
import datetime

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'sevir':
        """
            Load the data manually by using own external config
            :Config file = sevir_v1.yaml
            :return splitted train val and test dataloader

            note: The args value in main file doesnt affect the dataloader instead sevir_v1.yaml is the main config file
        """
        config_path =  "API/sevir_v1.yaml"  # Change to your project path
        oc_from_file = OmegaConf.load(open(config_path, "r"))
        dataset_oc = OmegaConf.to_object(oc_from_file.dataset)

        dm = get_sevir_datamodule(
                    dataset_oc=dataset_oc,
                    num_workers=8,)
        dm.prepare_data() # Check if SEVIR dataset is available
        dm.setup() # Preprocess train/val/test data set

        train_data_loader = dm.train_dataloader()
        val_data_loader = dm.val_dataloader()
        test_data_loader = dm.test_dataloader()
        print("data loading completed")
                

        return train_data_loader, val_data_loader, test_data_loader, 0,1
        