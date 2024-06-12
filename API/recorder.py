import numpy as np
import torch

class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score_val = None
        self.best_score_csi = None
        self.val_loss_min = np.Inf
        self.val_loss_csi_min = np.Inf
        self.delta = delta

    def __call__(self, epoch, val_loss, val_csi, model, path):
        score_mse = -val_loss
        score_csi = val_csi

        if self.best_score_val is None:
            self.best_score_val = score_mse
            self.save_checkpoint(epoch,val_loss, model, path, False)
        elif score_mse >= self.best_score_val + self.delta:
            self.best_score_val = score_mse
            self.save_checkpoint(epoch, val_loss, model, path, False)

        if self.best_score_csi is None:
            self.best_score_csi = score_csi
            self.save_checkpoint(epoch,val_csi, model, path , True)
        elif score_csi >= self.best_score_csi + self.delta:
            self.best_score_csi = score_csi
            self.save_checkpoint(epoch, val_csi, model, path ,True)

    def save_checkpoint(self, epoch, val_loss, model, path,bool_csi):
        if bool_csi:
            if self.verbose:
                print(f'Validation CSI increased ({self.val_loss_csi_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), f"{path}/exp/checkpoint_ep{epoch}_csi_loss{val_loss}.pth")
            torch.save(model.state_dict(), path+'/'+'checkpoint_csi.pth')
            self.val_loss_csi_min = val_loss
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), f"{path}/exp/checkpoint_ep{epoch}_mse_loss{val_loss}.pth")
            torch.save(model.state_dict(), path+'/'+'checkpoint_mse.pth')
            self.val_loss_min = val_loss