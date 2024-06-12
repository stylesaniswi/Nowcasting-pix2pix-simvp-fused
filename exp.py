
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from model import SimVP 
from tqdm import tqdm
from API import *
from utils import *
import torch.nn.functional as F
import pdb
import wandb
import torch.nn as nn
from API.preprocess_data import preprocess_data
from models.pix2pix.pix2pix_model import Pix2PixModel
from models.pix2pix.options import get_opt_pix2pix
from models.pix2pix.networks import UnetGenerator
import functools



class Exp:
    WANDB_LOGGING = True

    if WANDB_LOGGING:

        
        # wandb.init(project="simvp_only", name="all_data_hyperparameter_0.001")
        # wandb.init(project="sample_experiment_0.001")
        # wandb.init(project="sample_experiment_0.01", name="constant_hyperparameter_0.01")
        # wandb.init(project="pipeline_2")
        wandb.init(project="simvp_only", name="all_data_testing_01")

    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        
        self.fusion_model = self.args.fusion_model
        self.concat_input = self.args.concat_model

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()
        self._select_optimizer()
        self._select_criterion()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  #training going on here
            # os.environ["CUDA_VISIBLE_DEVICES"] = "2" # testing
            device = torch.device('cuda')
            print_log('Use GPUs: {}'.format(torch.cuda.device_count()))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    # def _build_model(self):
    #     args = self.args
    #     SimVP_model = SimVP(tuple(args.in_shape), args.hid_S,)
    #     # opt = get_opt_pix2pix()
    #     # Pix2pix = Pix2PixModel(opt)
    
    #     # net = UnetGenerator(args.input_nc, args.output_nc, args.num_downs, args.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout= False)

    #     # net.load_state_dict(torch.load('models/pix2pix/latest_net_G.pth'))
    #     # if torch.cuda.device_count() > 1:
    #     #     net = nn.DataParallel(net)
            
    #     # net.to(self.device)
        
    #     # # print("device of pix2pix" , next(net.module.parameters()).device)
    #     # self.net_model = net
    #     if torch.cuda.device_count() > 0:
    #         SimVP_model = nn.DataParallel(SimVP_model)

    #     self.model =SimVP_model.to(self.device)
    #     # pretrained_path = 'models/pix2pix/latest_net_G.pth'  # Path to your pretrained model file
    #     # pdb.set_trace()
    #     # Pix2pix.setup(pretrained_path)
    #     # pdb.set_trace()
    #     # if torch.cuda.device_count() > 1:
    #     #     Pix2pix = nn.DataParallel(Pix2pix)
            
    #     # self.Pix2pix =Pix2pix.to(self.device)
        
    #     # model = CombinedModel(,SimVP_model)
    #     # if torch.cuda.device_count() > 1:
    #     #     model = nn.DataParallel(model)
    #     # self.model =model.to(self.device)
    #     print("Model compiled successfully..")

    def _build_model(self):
        args = self.args
        SimVP_model = SimVP(tuple(args.in_shape), args.hid_S,)

        net = UnetGenerator(args.input_nc, args.output_nc, args.num_downs, args.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True), use_dropout= False)

        net.load_state_dict(torch.load('models/pix2pix/latest_net_G.pth'))
        if torch.cuda.device_count() > 0:
            net = nn.DataParallel(net)
            
        net.to(self.device)
        
        # print("device of pix2pix" , next(net.module.parameters()).device)
        self.net_model = net
        if torch.cuda.device_count() > 0:
            SimVP_model = nn.DataParallel(SimVP_model)

        self.model =SimVP_model.to(self.device)
       

        print("Model compiled successfully..")

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        # state = self.scheduler.state_dict()
        # fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        # pickle.dump(state, fw)
    
    def customModel(self, batch_x):
        """
            Fused model which takes satellite input and fetch to Pix2Pixe(net) model
            Generated Gen_Y is reshaped and sent to SimVP model for prediction
            :self.concat_input: Boolean value (If True concat the Gen_Y with satellite input on dim = 2 (channel Dim))
            :return: Predictive sequence from the model
        """
        B,T,C,H,W = batch_x.shape
        x_inp = batch_x.reshape(B* T, C, H, W)
        Gen_y =self.net_model(x_inp)
        pred_y = Gen_y.reshape(B,T,1,H,W)
        if self.concat_input:
            pred_y = torch.cat((batch_x, pred_y), dim=2)
        pred_y = self.model(pred_y)
                   
        pred_y = pred_y[:,:12]
        return pred_y

    def normalize_them(self, x, scale, offset, reverse=False):
        """
        Normalize data or reverse normalization
        :param x: data array
        :param scale: const scaling value
        :param offset: const offset value
        :param reverse: boolean undo normalization
        :return: normalized x array
        """
        if reverse:
            return x / scale - offset #actually +,  made - for unnormalizing after analyzing how it have been normalized while making dataset 
        else:
            return (x + offset) * scale


    
    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)


        for epoch in range(config['epochs']):
            print_log('Epoch: {}'.format(epoch + 1))
           
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)
            batch_idx =0 
            for batch in train_pbar:
                mode = 'onlysimvp' if self.fusion_model == False else 'fused'
                batch_x, batch_y = preprocess_data(batch, mode=mode ) #pipeline 2 : normalization with sevir for X only not for Y, choose mode = 'onlysimvp' for radar to radar prediction
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # pdb.set_trace()
                if self.fusion_model == False:
                    pred_y =self.model(batch_x)
                    pred_y = pred_y[:,:12] #for only simvp model

                #changes for pipeline 2
                else:    
                    pred_y = self.customModel(batch_x)
               
                loss = self.criterion(pred_y, batch_y)
              
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                
                
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                
                # current_lr = self.scheduler.get_last_lr()[0]
                current_lr = self.args.lr
                lr_log_dict ={
                    "Learning_rate": current_lr,
                    "Train_loss_lr" : train_loss[-1],
                    "Iteration" : batch_idx
                }
                print_log(f'Epoch [{epoch+1}/{self.args.epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], LR: {current_lr}')
                wandb.log(lr_log_dict)
                batch_idx +=1

            train_loss = np.average(train_loss)

            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss, vali_mean_csi = self.vali(self.vali_loader)
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))
                
                # current_lr = self.scheduler.get_last_lr()[0]
                current_lr = self.args.lr
                lr_log_dict ={
                    "Learning_rate_epochEnd": current_lr,
                    "val_loss" : vali_loss,
                    "val_csi" : vali_mean_csi
                }        
                
                wandb.log(lr_log_dict)
                log_dict = {
                    "Train_Loss": train_loss,
                    "Vali_Loss": vali_loss
                    }
                log_string = "Train Loss: {Train_Loss:.4f} | Vali Loss: {Vali_Loss:.4f}\n".format(**log_dict)
                print_log(f"Epoch: {epoch+1} " + log_string)
                wandb.log(log_dict)
                recorder(epoch+1, vali_loss, vali_mean_csi, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint_csi.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def  vali(self, vali_loader):
        self.model.eval()
        total_loss = []
        total_mse = 0
        total_mae = 0
        total_ssim = 0
        total_psnr = 0
        total_csi = 0
        total_csi181 = 0
        total_csi219 = 0
        total_samples = 0

        vali_pbar = tqdm(vali_loader)
        for i, batch in enumerate(vali_pbar):
            mode = 'onlysimvp' if self.fusion_model == False else 'fused'
            batch_x, batch_y = preprocess_data(batch, mode=mode) #pipeline 2 : normalization with sevir for X only not for Y

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            if self.fusion_model == False:
                pred_y =self.model(batch_x)
                pred_y = pred_y[:,:12] #for only simvp model

                #changes for pipeline 2
            else:    
                pred_y = self.customModel(batch_x)
            loss = self.criterion(pred_y, batch_y)
           
           
            """
                Unnormalizing before calculating the metrics
                Use it only when Ground Truth(Y) is normalized
                
                For Rescale_mode = "sevir" If wanted to normalize Y also then change the code on /API/sevir_dataloader.py value PREPROCESS_SCALE_SEVIR.. change it as written in comment.
                
                If rescale_mode in dataloader config = 'sevir' then use
                    pred_y  = self.normalize_them(pred_y ,1 / 47.54, -33.44, reverse =True)    
                    batch_y  = self.normalize_them(batch_y ,1 / 47.54, -33.44, reverse =True)
                Else if rescale_mode in dataloader config ='01' then use
                    pred_y  = self.normalize_them(pred_y ,1/255, 0, reverse =True)    
                    batch_y  = self.normalize_them(batch_y ,1/255, 0, reverse =True)
            """
            # pred_y  = self.normalize_them(pred_y ,1/255, 0, reverse =True)    
            # batch_y  = self.normalize_them(batch_y ,1/255, 0, reverse =True)
            """
                Till here ammended
            """
            

            mse, mae, ssim, psnr, csi, csi181, csi219 = metric(pred_y.detach().cpu().numpy(), batch_y.detach().cpu().numpy(), 0, 1, True)

            

            total_mse += mse.item()
            total_mae += mae.item()
            total_ssim += ssim.item()
            total_psnr += psnr.item()
            total_csi += csi.item()
            total_csi181 += csi181.item()
            total_csi219 += csi219.item()

            # Increment total number of samples
            total_samples += 1
            # pdb.set_trace()
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss.item())

        # pdb.set_trace()
        total_loss = np.average(total_loss)

        mean_mse = total_mse / total_samples
        mean_mae = total_mae / total_samples
        mean_ssim = total_ssim / total_samples
        mean_psnr = total_psnr / total_samples
        mean_csi = total_csi / total_samples
        mean_csi181 = total_csi181 / total_samples
        mean_csi219 = total_csi219 / total_samples
        # pdb.set_trace()        
        log_dict = {
            "vali_mse": mean_mse,
            "vali_mae": mean_mae,
            "vali_ssim": mean_ssim,
            "vali_psnr": mean_psnr,
            "vali_csi": mean_csi,
            "vali_csi181": mean_csi181,
            "vali_csi219": mean_csi219
            }
        log_string = 'vali mse:{vali_mse:.4f}, mae:{vali_mae:.4f}, ssim:{vali_ssim:.4f}, psnr:{vali_psnr:.4f}, csi:{vali_csi:.4f}, csi181:{vali_csi181:.4f},csi219:{vali_csi219:.4f}'.format(**log_dict)
        print_log(log_string)
        wandb.log(log_dict)
        self.model.train()
        return total_loss , mean_csi

    def test(self, args):
        self.model.load_state_dict(torch.load("results/onlysimvp/checkpoint_csi.pth"))
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        total_mse = 0
        total_mae = 0
        total_ssim = 0
        total_psnr = 0
        total_csi = 0
        total_csi181 = 0
        total_csi219 = 0
        total_samples = 0
        print_log('Each batch loss: ')
        test_pbar = tqdm(self.test_loader)  
        for idx, batch in enumerate(test_pbar):
            mode = 'onlysimvp' if self.fusion_model == False else 'fused'
            batch_x, batch_y = preprocess_data(batch, mode=mode)
            if self.fusion_model == False:
                pred_y =self.model(batch_x)
                pred_y = pred_y[:,:12] #for only simvp model

                #changes for pipeline 2
            else:    
                pred_y = self.customModel(batch_x)
                
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))
            
            """
                Use it only when Ground Truth(Y) is normalized.
                
                For Rescale_mode = "sevir" If wanted to normalize Y also then change the code on /API/sevir_dataloader.py value PREPROCESS_SCALE_SEVIR.. change it as written in comment.
                
                If rescale_mode in dataloader config = 'sevir' then use
                    pred_y  = self.normalize_them(pred_y ,1 / 47.54, -33.44, reverse =True)    
                    batch_y  = self.normalize_them(batch_y ,1 / 47.54, -33.44, reverse =True)
                Else if rescale_mode in dataloader config ='01' then use
                    pred_y  = self.normalize_them(pred_y ,1/255, 0, reverse =True)    
                    batch_y  = self.normalize_them(batch_y ,1/255, 0, reverse =True)
            """
            # pred_y  = self.normalize_them(pred_y ,1/255, 0, reverse =True)    
            # batch_y  = self.normalize_them(batch_y ,1/255, 0, reverse =True)
            """
                Till here ammended
            """
            mse, mae, ssim, psnr, csi, csi181, csi219 = metric(pred_y.detach().cpu().numpy(), batch_y.detach().cpu().numpy(), 0, 1, True)
            log_dict = {
                "test_mse": mse,
                "test_mae": mae,
                "test_ssim": ssim,
                "test_psnr": psnr,
                "test_csi": csi,
                "test_csi181": csi181,
                "test_csi219": csi219
            }
            log_string = 'Test(batchwise) mse:{test_mse:.4f}, mae:{test_mae:.4f}, ssim:{test_ssim:.4f}, psnr:{test_psnr:.4f}, csi:{test_csi:.4f}, csi181:{test_csi181:.4f}, csi219:{test_csi219:.4f}'.format(**log_dict)
            print_log(log_string)
            wandb.log(log_dict)

            # total_samples += batch_y.size(0)
            total_samples += 1

            total_mse += mse.item()
            total_mae += mae.item()
            total_ssim += ssim.item()
            total_psnr += psnr.item()
            total_csi += csi.item()
            total_csi181 += csi181.item()
            total_csi219 += csi219.item()

        

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])
        
        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

 
        print_log('Total Testing loss ------------ \n')
        mean_mse = total_mse / total_samples
        mean_mae = total_mae / total_samples
        mean_ssim = total_ssim / total_samples
        mean_psnr = total_psnr / total_samples
        mean_csi = total_csi / total_samples
        mean_csi181 = total_csi181 / total_samples
        mean_csi219 = total_csi219 / total_samples
        # pdb.set_trace()        
        log_dict = {
            "test_mse": mean_mse,
            "test_mae": mean_mae,
            "test_ssim": mean_ssim,
            "test_psnr": mean_psnr,
            "test_csi": mean_csi,
            "test_csi181": mean_csi181,
            "test_csi219": mean_csi219
            }
        log_string = 'Total Test loss (Average) :  mse:{test_mse:.4f}, mae:{test_mae:.4f}, ssim:{test_ssim:.4f}, psnr:{test_psnr:.4f}, csi:{test_csi:.4f}, csi181:{test_csi181:.4f},csi219:{test_csi219:.4f}'.format(**log_dict)
        print_log(log_string)

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])

        return mse
    