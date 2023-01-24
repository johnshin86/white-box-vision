import torch
import torch.distributed as dist

from contextlib import contextmanager
from dataclasses import dataclass, field 
from itertools import starmap, zip_longest
from typing import Callable, Generator, Iterable, Optional, overload, Type, Union

from .model_surgery import get_transformer_layers




# functions that generate and store intermediate representations
# of a vision model. 


@dataclass
class ResidualStream:

	embeddings: Optional[torch.Tensor] = None
	attentions: list[torch.Tensor] = field(default_factory = list)
	layers: list[torch.Tensor] = field(default_factory = list)


@contextmanager
def record_residual_stream(
	model: torch.nn.Module,
	*,
	include_input: bool = True,
	norm_class: Type[torch.nn.Module] = torch.nn.LayerNorm,
	post_norm: bool = False,
	retain_grads: bool = False,
	sublayers: bool = False,
	) -> Generator[ResidualStream, None, None]:
	"""Record every state of the residual stream in a transformer forward pass.

	This is a context manager that adds forward hooks to each `nn.LayerNorm`
	module in the transformer, storing the output in a dictionary keyed by the 
	layer norm's name. This dictionary is yielded by the context manager for analysis.

	Parameters
	----------
	model: torch.nn.Module
		
		Torch model for which to record the residual stream.

	include_input: bool
		
		Whether to record the input embeddings.

	norm_class: torch.nn.Module
		
		Pytorch normalizing layer.

	post_norm: bool

		Whether the transformer uses LayerNorm after residual connections.
	
	retain_grads:

		Whether to retain gradients for recorded states.

	sublayers:

		Whether to record attention and layer outputs separately. 


	Returns
	-------
	"""
	hooks = []
	residual_stream = ResidualStream()

	def process(state: torch.Tensor) -> torch.Tensor:
		if retain_grads:
			state.retain_grads_(True)
			state.retain_grad()

		else:
			state = state.detach()

		return state 

	def store_embeddings(module: torch.nn.Module, inputs) -> None:
		residual_stream.embeddings = process(inputs[0])

	def store_attention(module: torch.nn.Module, inputs) -> None:
		residual_stream.attentions.append(process(inputs[0]))

	def store_layer(module: torch.nn.Module, inputs, output: Union[torch.Tensor, tuple]):
		# Huggingface layers typically return tuples
		if isinstance(output, tuple):
			output = output[0]
			if not isinstance(output, torch.Tensor):
				idx = len(residual_stream.layers)
				raise RuntimeError(f"Expected first element of layer {idx} output to be a Tensor")

		residual_stream.layers.append(output)

	_, layers = get_model_layers(model)

	#Since embedding layer is not in ModuleList, we register a prehook on the input
	if include_input:
		hooks.append(layers[0].register_forward_pre_hook(store_embeddings))

	for i, layer in enumerate(layers):
		hooks.append(layer.register_forward_hook(store_layer))

		#Looking for inner layernorm
		if sublayers:
			layer_norms = [m for m in layer.modules() if isinstance(m, norm_class)]
			if not layer_norms:
				if sublayers:
					raise ValueError(
						f"No LNs found in layer {i}; try specifying `norm_class`"
					)
				else:
					continue
			elif len(layer_norms) != 2:
				if sublayers:
					raise ValueError(
						f"Expected 2 layer norms per layer when sublayers=True,"
						f"but layer {i} has {len(layer_norms)}"
					)

				else:
					continue

			#Get the post attention layer norm, account for
			#whether layernorm is applied before or after residual
			post_attention_ln = layer_norms[0 if post_norm else 1]
			hooks.append(post_attention_ln.register_forward_pre_hook(store_attention))

	try:
		yield residual_stream

	finally:
		for hook in hooks:
			hook.remove()








