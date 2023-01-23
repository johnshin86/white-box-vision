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

	Parameters
	----------
	model
	include_input
	norm_class
	post_norm
	retain_grads
	sublayers



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
		if isinstance(output, tuple):
			output = output[0]
			if not isinstance(output, torch.Tensor):
				idx = len(residual_stream.layers)
				raise RuntimeError(f"Expected first element of layer {idx} output to be a Tensor")

		residual_stream.layers.append(output)

	_, layers = get_model_layers(model)

	if include_input:
		hooks.append(layer[0].register_forward_hook(store_layer))

	for i, layer in enumerate(layers):
		hooks.append(layer.register_forward_hook(store_layer))

		if sublayers:
			layer_norms = [m for m in layer.modules() if isinstance(m, norm_class)]
			if not layer_norms:
				if sublayers:
					raise ValueError(
						f"No LNs found in layer {i}; try specifying `norm_class"
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

			post_attention_ln = layer_norms[0 if post_norm else 1]
			hooks.append(post_attention_ln.register_forward_hook(store_attention))

	try:
		yield residual_stream

	finally:
		for hook in hooks:
			hook.remove()








