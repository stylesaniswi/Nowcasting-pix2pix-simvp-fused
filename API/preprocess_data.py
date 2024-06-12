import torch
import torch.nn.functional as F
from normalize import zscore_normalizations

def preprocess_sevir(X,Y):
    """
        Resize the samples in to radar image size (384*384) for all the X satellite input and Y radar also

        :input : Takes list of X and Y
        :returns : Resized and stacked X inputs making tensor shape (B,13,C,384,384), C = no. of channels, In this case 3 as we have ir069, ir107, vis input channels
                    and Y ground truth tensor of shape (B,12,1,384,384)
    """
    max_height = 384
    max_width = 384

    # X = [x.squeeze(4) for x in X]
    # Y = [y.squeeze(4) for y in Y]
    X_resized=[]
    for x in X:
        resized_sequences = []
        for i in range(x.size(0)):
            sequence = x[i]  # Get the ith sequence

            resized_sequence = F.interpolate(sequence, size=(max_height, max_width), mode='bicubic', align_corners=False)
            resized_sequences.append(resized_sequence)
               # Stack the resized sequences back into a tensor
        resized_tensor = torch.stack(resized_sequences)
        X_resized.append(resized_tensor)
           
        
               

    Y_resized=[]
    for y in Y:
        resized_sequences = []
        for i in range(y.size(0)):
            sequence = y[i]  # Get the ith sequence
            resized_sequence = F.interpolate(sequence, size=(max_height, max_width), mode='bicubic', align_corners=False)
            resized_sequences.append(resized_sequence)

                # Stack the resized sequences back into a tensor
        resized_tensor = torch.stack(resized_sequences)
        Y_resized.append(resized_tensor)
        # X_resized = [F.interpolate(x, size=(1,max_height, max_width), mode='bicubic') for x in X]
        # Y_resized = [F.interpolate(y, size=(max_height, max_width),mode='bicubic') for y in Y]

        # Stack resized tensors along the batch dimension
    batch_x = torch.cat(X_resized,dim=2)
    batch_y=  torch.cat(Y_resized,dim=2)


    return batch_x, batch_y

def process_pix2pix_output(x,y):
    x_y =[x,y]
    return torch.cat(x_y, dim=2)

def normalize_them(x, scale, offset, reverse=False):
    """
    Normalize data or reverse normalization
    :param x: data array
    :param scale: const scaling value
    :param offset: const offset value
    :param reverse: boolean undo normalization
    :return: normalized x array
    """
    if reverse:
        return x / scale + offset
    else:
        return (x-offset) * scale

def preprocess_data(dataloader):
    """
        If the mode is onlysimvp then use: 
            :input = First 13 sequence of radar channel
            :returns [X,Y] = first 13 sequence of radar channel and last 12 sequence of radar channel in list respectively
        
        else if mode is fused:
            :input = first 13 sequence of input X satellite channels ( ir069, ir107 and Vis)
            : returns [X,Y] = list of first 13 sequence of 3 satellite channel , list of last 12 sequence of 1 radar channel
    """
    mode ='onlysimvp' # please write the mode  manually here 
    if mode == 'onlysimvp':
        vil= dataloader['vil']
        vil_12 = vil[:,13:] # after 13 sequence for ground truth
        #First 13 sequence for input
        vil_13 =vil[:,:13]
        X=[vil_13]
        Y=[vil_12]
        return preprocess_sevir(X,Y)
    
    else:
        vil= dataloader['vil']
        ir069= dataloader['ir069']
        ir107= dataloader['ir107']
        vis= dataloader['vis']

        vil_12 = vil[:,13:] # after 13 sequence for ground truth
        #First 13 sequence for input
        ir069_13 = ir069[:,:13]
        ir107_13 = ir107[:,:13]
        vis_13 =vis[:,:13]
        
        X=[ir069_13,ir107_13,vis_13]
        Y=[vil_12]

        return preprocess_sevir(X,Y)    
