import torch
import timm
from typing import Any, Generator, Optional, Type, TypeVar, Union



from timm.models.vision_transformer import Block 

# pretrained models will be from huggingface, 
# since afaik, it's the most popular NN arch library. 

# Suite of functions that will add / remove / replace / retrieve layers
# from a vision model.

# We want to be able to retrieve attention masks from the vision transformer. 
# As well as intermediate representations of the token(s). 

# We can either monkey patch the attention and block modules
# to return attention, or we can create subclasses of attention
# and block, which override the method. 

# Accessing specific layers in a pytorch model is a little wonky
# since it requires typing attributes in a chain. Sometimes
# the attributes are numbers, and sometimes they are strings.

#Helper functions to retrieve NN layers using strings. 

#Generator[yield_type, send_type, return_type]


# Functions to retrieve layers with strings

def get_value(obj: Any, key: str) -> Any:
	return obj[int(key)] if key.isdigit() else getattr(obj, key)

def set_value(obj: Any, key: str, value: Any) -> None:
	if key.isdigit():
		obj[int(key)] = value 
	else:
		setattr(obj, key, value)

def get_value_with_key_path(model: torch.nn.Module, key_path: str) -> Any:
	#iterate over key_path and retrieve values
	for key in key_path.split("."):
		model = get_value(model, key)

	return model

def set_value_with_key_path(
	model: torch.nn.Module, 
	key_path: str, 
	value: Union[torch.nn.Module, torch.Tensor]) -> None:

	keys = ket_path.split(".")

	#move to penultimate layer in key_path
	for key in keys[:-1]:
		model = get_value(model, key)

	#replace penultimate layer with value
	setattr(model, keys[-1], value)

T = TypeVar("T", bound = torch.nn.Module)



def get_model_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
	"""Get model layers from a model. Need to think about how to best handle this
	for vision models.
	"""
	return None


# Functions to manipulate a layer

@contextmanager
def set_layer(model: T, key_path: str, value: Any) -> Generator[T, None, None]:
	r"""Temporarily set a value by key path while in context.
	"""
	old_value = get_value_with_key_path(model, key_path)
	set_value_with_key_path(model, key_path, value)

	try:
		yield model 
	finally:
		set_value_with_key_path(model, key_path, old_value)

@contextmanager
def delete_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
	list_path, layer_list = get_model_layers(model)
	modified_list = torch.nn.ModuleList(layer_list)

	for i in sorted(indices, reverse=True):
		del modified_list[i]

	set_value_with_key_path(model, list_path, modified_list)

	try:
		yield model

	finally:
		set_value_with_key_path(model, list_path, layer_list)

@contextmanager
def permute_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
	list_path, layer_list = get_model_layers(model)
	permuted_list = torch.nn.ModuleList([layer_list[i] for i in indices])

	set_value_with_key_path(model, list_path, permuted_list)

	try:
		yield model 
	finally:
		set_value_with_key_path(model, list_path, layer_list)


@contextmanager
def replace_layers(model: T, indices: list[int], replacements: list[torch.nn.Module]) -> Generator[T, None, None]:
	
	assert len(indices) == len(replacements), "List of indices should be same length as replacements."

	list_path, layer_list = get_model_layers(model)
	modified_list = torch.nn.ModuleList(layer_list)

	for i, replacement in zip(indices, replacements):
		modified_list[i] = replacement  

	set_value_with_key_path(model, list_path, modified_list)

	try:
		yield model 

	finally:
		set_value_with_key_path(model, list_path, layer_list)










