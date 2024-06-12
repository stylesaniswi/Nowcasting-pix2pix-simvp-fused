#Use another metrics.py file feom utils folder.


# import numpy as np
# from skimage.metrics import structural_similarity as cal_ssim
# import pdb
# import torch
# from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
# from torchmetrics.regression import CriticalSuccessIndex


# # def MAE(pred, true):
# #     pdb.set_trace()
# #     return np.mean(np.abs(pred-true),axis=(0,1)).sum()

# # def MSE(pred, true):
# #     return np.mean((pred-true)**2,axis=(0,1)).sum()

# # cite the `PSNR` code from E3d-LSTM, Thanks!
# # https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
# def PSNR(pred, true):
#     mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
#     return 20 * np.log10(255) - 10 * np.log10(mse)

# def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
#     """
#         Added metrics evaluation method from torch.metrics
#         :returns the calculated MAE, MSE, CSI for threshold 181,219 and the average of 6 thresholds.
#     """
#     pred_tensor = torch.from_numpy(pred)
#     true_tensor = torch.from_numpy(true)
#     # pred = pred*std + mean
#     # true = true*std + mean
#     # pdb.set_trace()
#     MAE_loss = torch.nn.L1Loss()
#     MSE_loss = torch.nn.MSELoss()
#     thresholds = [16, 74, 133, 160, 181, 219]
#     csis =[]
#     csi181 = 0
#     csi219 =0
#     for i in thresholds:
#         csi_loss = CriticalSuccessIndex(threshold=i)
#         csi_each_thres = csi_loss(pred_tensor , true_tensor)
#         if i == 181:
#             csi181 = csi_each_thres
#         if i==219:
#             csi219 = csi_each_thres
#         csis.append(csi_each_thres)
#     # pdb.set_trace()
#     csis_concatenated = torch.cat([csi_tensor.unsqueeze(0) for csi_tensor in csis], dim=0)
 
#     mcsi = torch.mean(csis_concatenated)

#     mae = MAE_loss(pred_tensor, true_tensor)
#     mse = MSE_loss(pred_tensor, true_tensor)
    

#     if return_ssim_psnr:

#         ssim, psnr = 0, 0
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 ssim += cal_ssim(pred[b, f,0], true[b, f,0],data_range=max(np.max(true[b,f,0]) - np.min(true[b,f,0]) , np.max(pred[b,f,0]) - np.min(pred[b,f,0])), multichannel=False)
#                 psnr += PSNR(pred[b, f,0], true[b, f,0])
#         ssim = ssim / (pred.shape[0] * pred.shape[1])
#         psnr = psnr / (pred.shape[0] * pred.shape[1])
#         return mse, mae, ssim, psnr, mcsi, csi181, csi219
#     else:
#         return mse, mae