import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')
import wandb

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--name', default="latest", type=str)
    # parser.add_argument('--gpu', default=5, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='sevir', choices=['mmnist', 'taxibj', 'sevir'])
    parser.add_argument('--num_workers', default=0, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[13, 1, 384, 384], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj, [13, 1, 384, 384] for only radarnowcasting, [13, 4, 384, 384] for fused custom model taking 3input and 1output from inference model
    parser.add_argument('--hid_S', default=64, type=int)
    # parser.add_argument('--hid_T', default=512, type=int) #initially 256
    parser.add_argument('--hid_T', default=256, type=int) #initially 256
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)
    
    #model parameters for Pix2pix pretrained model
    parser.add_argument('--input_nc',default= 3, type =int )
    parser.add_argument('--output_nc',default= 1, type =int )
    parser.add_argument('--ngf',default= 64, type =int )
    parser.add_argument('--ndf',default= 64, type =int )
    parser.add_argument('--num_downs',default= 7, type =int )

    # Training parameters
    parser.add_argument('--fusion_model',default= False, type =bool, help="If True it will fuse pix2pix model as inference and simvp as predictive" )
    parser.add_argument('--concat_input',default= False, type =bool, help="If True Concat satellite input with pix2pix generated radar image , See pipelines" )
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default= 0.01, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    wandb.init(project="simvp_only")
    # wandb.init(project="main_project_2")
    # wandb.init(project="pipeline_2") # for 3 input pix2 pix 1 input predicted from pix2pix for simvp
    # wandb.init(project="sample_experiment")
    # wandb.init(project="sample_experiment_0.001")
    # wandb.init(project="sample_experiment_0.01")


    exp = Exp(args)
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
    wandb.finish()
