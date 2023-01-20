import torch
import timm
from typing import Any, Generator, Optional, Type, TypeVar, Union



from timm.models.vision_transformer import Block 

# pretrained models will be from timm, 
# since afaik, it's the most popular NN arch library for vision. 

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

@contextmanager
def tmp_value_with_key_path(model: T, key_path: str, value: Any) -> Generator[T, None, None]:
	r"""Temporarily set a value by key path while in context.
	"""
	old_value = get_value_with_key_path(model, key_path)
	set_value_with_key_path(model, key_path, value)

	try:
		yield model 
	finally:
		set_value_with_key_path(model, key_path, old_value)


def get_model_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
	"""Get model layers from a model.
	"""



