import torch
import timm


from timm.models.vision_transformer import Block 

# pretrained models will be from timm, 
# since afaik, it's the most popular NN arch library. 

# Suite of functions that will add / remove / replace / retrieve layers
# from a vision model.

# We want to be able to retrieve attention masks from the vision transformer. 
# As well as intermediate representations of the token(s). 

# We can either monkey patch the attention and block modules
# to return attention, or we can create subclasses of attention
# and block, which override the method. 

# Accessing specific layers in a pytorch model is a little wonky
# since it requires typing attributes in a chain.
# so we need functions to manipulate this. 

