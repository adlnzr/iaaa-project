import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

'''
there are two main ViT concepts in the model architecture

1- VisionTransformerSPT_LSA 
input: iamges of each patient (sequence of patches) -> 
output: CLS of each image 

model: (ShiftedPatchTokenization + 12-layer transformer encoder layer + LocalitySelfAttention)

2- StandardViT
input: stacked CLS's of each patient (sequence of CLS's) ->  
output: CLS of each patient -> classifier

model: (12-layer transformer encoder layer)
'''

class ShiftedPatchTokenization(nn.Module):
    '''
    as the final patch vector sizes are 768 -> so nn.Conv2d is used to map the 
    images (+ shiftted images) into 768 size vectors
    '''

    def __init__(self, patch_size=16, in_channels=1, hidden_dim=768, shift_size=4):
        super(ShiftedPatchTokenization, self).__init__()
        self.patch_size = patch_size
        self.shift_size = shift_size
        self.projection = nn.Conv2d(in_channels * (shift_size + 1), hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        shifted_images = [x]
        
        # Apply shifts in four diagonal directions
        for direction in [(-self.shift_size, -self.shift_size), (self.shift_size, self.shift_size),
                          (-self.shift_size, self.shift_size), (self.shift_size, -self.shift_size)]:
            shifted_x = torch.roll(x, shifts=direction, dims=(2, 3))
            shifted_images.append(shifted_x)
        
        # Concatenate the original and shifted images along the channel dimension
        x = torch.cat(shifted_images, dim=1) # shape: [4, 15, 224, 224]

        # Apply the patch embedding projection
        x = self.projection(x)  # shape: [B, 768, 14, 14]

        x = x.flatten(2).transpose(1, 2)  # shape: [B, N_patches, hidden_dim]
        return x

# Locality Self-Attention (LSA)
class LocalitySelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(LocalitySelfAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute the attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn - torch.diag_embed(torch.full((N,), float('-1e9'), device=attn.device)).unsqueeze(0).unsqueeze(0)  # Diagonal masking
        attn = self.softmax(attn / self.temperature)  # Learnable temperature scaling
        attn = self.attn_drop(attn)

        # Compute the output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# Vision Transformer Model with SPT and LSA
class VisionTransformerSPT_LSA(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, hidden_dim=768, num_heads=12, 
                 num_layers=12, vit_model = 'google/vit-base-patch16-224-in21k'):
        super(VisionTransformerSPT_LSA, self).__init__()
        
        # Shifted Patch Tokenization
        self.spt = ShiftedPatchTokenization(patch_size=patch_size, in_channels=in_channels, hidden_dim=hidden_dim)
        
        # load the configuration for the ViT model
        # self.config = ViTConfig.from_pretrained(vit_model)

        # transformer layers (12 layers)
        # self.transformer_model = ViTModel(self.config)

        # load a pretrained ViT model
        self.transformer_model = ViTModel.from_pretrained(vit_model)
        
        # locality Self-Attention
        self.lsa = LocalitySelfAttention(hidden_dim, num_heads) 

    def forward(self, x):
        # Step 1: Tokenize the image using SPT shape: [bs, 1, 224, 224]
        x = self.spt(x) #  shape: [bs, 196, 768]
        
        # Step 2: Pass through Transformer layers with LSA
        x = self.transformer_model.encoder(x) # shape: [bs, 196, 768]
        
        x = self.lsa(x.last_hidden_state) # shape: [bs, 196, 768]

        return x[:,0] # the CLS token to be stacked for the next ViT model 
        

class VisionTransformerForTokens(nn.Module):
    def __init__(self, hidden_dim = 768, num_classes = 1, vit_model = 'google/vit-base-patch16-224-in21k'):
        super().__init__()

        self.vit_model = vit_model

        # load the configuration for the ViT model
        # self.config = ViTConfig.from_pretrained(vit_model)

        # transformer layers (12 layers)
        # self.transformer_model = ViTModel(self.config)

        # laod a pre-trained ViT model
        self.transformer_model = ViTModel.from_pretrained(vit_model)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        '''
        CLS token is added to the input which is stacked of CLS tokens belongs to each patient

        positional encoding is not added to the input :
        each token of the input is extracted from an image of the patient so as supposed the images of each patient 
        do not follow a particular order the positional encoding would not be helful
        '''
        # add the CLS token
        cls_token = torch.zeros(x.size(0), 1, x.size(-1), device=x.device)  # shape: [batch_size, 1, 768]
        x = torch.cat((cls_token, x), dim=1)  # Now x is [batch_size, 21, 768] (20 patches + 1 CLS token)

        # path the data through 12-layer transformer encoder
        x = self.transformer_model.encoder(x)
        cls_token = x.last_hidden_state[:, 0]
        logits = self.mlp_head(cls_token)
        return logits
    
