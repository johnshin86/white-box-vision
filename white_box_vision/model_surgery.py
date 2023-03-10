import torch
import timm
from contextlib import contextmanager
from typing import Any, Generator, Optional, Type, TypeVar, Union

# pretrained models will be from huggingface, 
# since afaik, it's the most popular NN arch library. 

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

# Shall we have different functions for huggingface vs timm?

def get_model_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
	"""Get model layers from a model. This looks for a ModuleList which contains
	the majority of the parameters. This will not include embeddings and the
	classification layer. 
	"""
	total_params = sum(p.numel() for p in model.parameters())
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.ModuleList):
			module_params = sum(p.numel() for p in module.parameters())

			if module_params > total_params / 2:
				return name, module

	raise ValueError(
		"Could not find `ModuleList` of at least half the model parameters."
	)

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










