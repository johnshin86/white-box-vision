import torch
import timm

from timm.models.vision_transformer import Block 

# pretrained models will be from timm, 
# since afaik, it's the most popular ViT library. 

# Suite of functions that will add / remove / replace / retrieve layers
# from a Vision Transformer.

# The basic vision transformer architecture is -
# Attention module
# LayerScale module
# Block module with Attention, Layer Scale, DropPath, MLP, Layer Scale, Drop Path

# We want to be able to retrieve attention masks from the vision transformer. 
# As well as intermediate representations of the class token. 

# We can either monkey patch the attention and block modules
# to return attention, or we can create subclasses of attention
# and block, which override the method. 

def forward_with_attention(self, x, return_attention = False):
	x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
	x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
	return x