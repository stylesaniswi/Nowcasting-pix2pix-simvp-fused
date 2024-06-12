# import torch
# from torch import nn
# from modules import ConvSC, Inception
# from API.preprocess_data import process_pix2pix_output
# import pdb

# def stride_generator(N, reverse=False):
#     strides = [1, 2]*10
#     if reverse: return list(reversed(strides[:N]))
#     else: return strides[:N]

# class Encoder(nn.Module):
#     def __init__(self,C_in, C_hid, N_S):
#         super(Encoder,self).__init__()
#         strides = stride_generator(N_S)
#         self.enc = nn.Sequential(
#             ConvSC(C_in, C_hid, stride=strides[0]),
#             *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
#         )
    
#     def forward(self,x):# B*4, 3, 128, 128
#         enc1 = self.enc[0](x)
#         latent = enc1
#         for i in range(1,len(self.enc)):
#             latent = self.enc[i](latent)
#         return latent,enc1


# class Decoder(nn.Module):
#     def __init__(self,C_hid, C_out, N_S):
#         super(Decoder,self).__init__()
#         strides = stride_generator(N_S, reverse=True)
#         self.dec = nn.Sequential(
#             *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
#             ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
#         )
#         self.readout = nn.Conv2d(C_hid, C_out, 1)
    
#     def forward(self, hid, enc1=None):
#         for i in range(0,len(self.dec)-1):
#             hid = self.dec[i](hid)
#         Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
#         Y = self.readout(Y)
#         return Y

# class Mid_Xnet(nn.Module):
#     def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
#         super(Mid_Xnet, self).__init__()

#         self.N_T = N_T
#         enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
#         for i in range(1, N_T-1):
#             enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
#         enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

#         dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
#         for i in range(1, N_T-1):
#             dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
#         dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

#         self.enc = nn.Sequential(*enc_layers)
#         self.dec = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.reshape(B, T*C, H, W)

#         # encoder
#         skips = []
#         z = x
#         for i in range(self.N_T):
#             z = self.enc[i](z)
#             if i < self.N_T - 1:
#                 skips.append(z)

#         # decoder
#         z = self.dec[0](z)
#         for i in range(1, self.N_T):
#             z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

#         y = z.reshape(B, T, C, H, W)
#         return y


# class SimVP(nn.Module):
#     def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
#         super(SimVP, self).__init__()
#         T, C, H, W = shape_in
#         self.enc = Encoder(C, hid_S, N_S)
#         self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
#         self.dec = Decoder(hid_S, 1, N_S)


#     def forward(self, x_raw):
#         B, T, C, H, W = x_raw.shape
#         x = x_raw.view(B*T, C, H, W)

#         embed, skip = self.enc(x)
#         _, C_, H_, W_ = embed.shape

#         z = embed.view(B, T, C_, H_, W_)
#         hid = self.hid(z)
#         hid = hid.reshape(B*T, C_, H_, W_)

#         Y = self.dec(hid, skip)
#         Y = Y.reshape(B, T, 1, H, W)
#         return Y

# # class SimVP(nn.Module):
# #     def __init__(self, shape_in, hid_S=16, incep_ker=[3, 5, 7, 11], groups=8):
# #         super(SimVP, self).__init__()
# #         C, H, W = shape_in
# #         self.enc = Encoder(C, hid_S, 4)
# #         self.dec = Decoder(hid_S, 1, 4)

# #     def forward(self, x_raw):
        
# #         B, C, H, W = x_raw.shape
# #         x = x_raw.view(B, C, H, W)

# #         embed, skip = self.enc(x)
# #         _, C_, H_, W_ = embed.shape

# #         hid = embed.view(B, C_, H_, W_)
# #         Y = self.dec(hid, skip)
# #         Y= Y.reshape(B, 1, H, W)
# #         return Y

# class CombinedModel(nn.Module):
#     def __init__(self, pix2pix_model, simvp_model):
#         super(CombinedModel, self).__init__()
#         self.pix2pix = pix2pix_model
#         self.simvp = simvp_model
#         self.device= torch.device('cuda')

#     def forward(self, x):
#         # Forward pass through Pix2Pix model
        
#         self.pix2pix.eval()  # Set model to evaluation mode

#         # Assuming 'input_data' contains your input data
#         B,T,C,H,W = x.shape
#         # pdb.set_trace()

#         x_inp = x.reshape(B* T, C, H, W)
#         self.pix2pix.module.set_input(x_inp)
#         self.pix2pix.module.forward()  # Run forward pass to generate output

#         # Access the generated output
#         # pdb.set_trace()
#         y_pix2pix = self.pix2pix.module.fake_B
#         # print(y_pix2pix.shape)
#         y_pix2pix = y_pix2pix.reshape(B,T,1,H,W)


#         # Process Pix2Pix output and send it to SimVP
#         # pdb.set_trace()
#         cat_inp =torch.cat((x, y_pix2pix.to(self.device)), dim=2)
#         print(cat_inp)
#         print(cat_inp.shape)
#         # cat_inp = process_pix2pix_output(x,y_pix2pix)  # Implement this function
#         # pdb.set_trace()
#         # Forward pass through SimVP model
#         # simvp_output = self.simvp(cat_inp)
        
#         return cat_inp


import torch
from torch import nn
from modules import ConvSC, Inception

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y