import torch
import torch.distributed as dist

from contextlib import contextmanager
from dataclasses import dataclass, field 
from itertools import starmap, zip_longest
from typing import Callable, Generator, Iterable, Optional, overload, Type, Union

from .model_surgery import get_model_layers

# functions that generate and store intermediate representations
# of a vision model. 

@dataclass
class ResidualStream:
	"""Data class of collection of transformer hidden states and derived quantities.
	"""

	embeddings: Optional[torch.Tensor] = None
	attentions: list[torch.Tensor] = field(default_factory = list)
	layers: list[torch.Tensor] = field(default_factory = list)

	@classmethod
	def stack(cls, streams: list["ResidualStream"]) -> "ResidualStream":
		"""Stack a list of residual streams into a single stream.
		"""
		if len(streams) < 2:
			raise ValueError("Expected at least two streams.")

		first, *rest = streams
		return first.zip_map(lambda *tensors: torch.stack(tensors), *rest)

	def all_reduce_(self):
		"""All-reduce all states over distributed workers."""
		if not dist.is_initialized():
			return

		for state in self:
			dist.all_reduce(state)
			state /= dist.get_world_size()

	def clear(self) -> None:
		"""Clear all residual states.
		"""
		self.embeddings = None
		self.attentions.clear()
		self.layers.clear()

	@property
	def has_sublayers(self) -> bool:
		"""Return whether the stream contains both attention and layer outputs."""
		return bool(self.attentions) and bool(self.layers)

	@property
	def shape(self):
		"""Return the shape of the first state.
		"""
		return next(iter(self)).shape
	
	def items(
		self, reverse: bool = False
	) -> Generator[tuple[str, torch.Tensor], None, None]:
		"""Iterate over residual states in topological order, yielding labels.
		"""
		if not reverse:
			if self.embeddings is not None:
				yield "input", self.embeddings 

			for i, (attention, layer) in enumerate(
				zip_longest(self.attentions, self.layers)
			):

				if attention is not None:
					yield f"{i}.attention", attention
				if layer is not None:
					yield f"{i}.ffn", layer

		else:
			attentions, layers = reversed(self.attentions), reversed(self.layers)
			for i, (attention, layer) in enumerate(zip_longest(attentions, layers)):
				if layer is not None:
					yield f"{i}.ffn", layer 

				if attention is not None:
					yield f"{i}.attention", attention 

			if self.embeddings is not None:
				yield "input", self.embeddings

	def labels(self) -> list[str]:
		"""Return labels for the residual states suitable for plotting.
		"""
		return [label for label, _ in self.items()]

	def map(self, fn: Callable, reverse: bool = False) -> "ResidualStream":
		"""Map a function over all states, returning a new `ResidualStream`.
		"""
		it = reversed(self) if reverse else iter(self)
		return self.new_from_list(list(map(fn, it)))

	def pairwise_map(
		self, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
	) -> "ResidualStream":
		"""Map over adjacent pairs of states, returning a new `ResidualStream.`
		"""
		if self.embeddings is None:
			raise ValueError("Can't map pairwise without input embeddings.")

		states = list(self)
		return self.new_from_list(list(starmap(fn, zip(states[:-1], states[1:]))))

	def zip_map(self, fn: Callable, *others: "Iterable") -> "ResidualStream":
		"""Map over corresponding states, returning a new `ResidualStream`.
		"""
		return self.new_from_list(list(starmap(fn, zip(self, *others))))

	def new_from_list(self, states: list[torch.Tensor]) -> "ResidualStream":
		"""Create a new `ResidualStream` with the given states.
		"""
		embeddings = states.pop(0) if self.embeddings is not None else None

		if self.has_sublayers:
			return ResidualStream(
				embeddings=embeddings, attentions=states[::2], layers=states[1::2]
			)

		else:
			return ResidualStream(embeddings=embeddings, layers=states)

	def plot(self, tick_spacing: int = 2, **kwargs):
		"""Plot the residual states.
		"""
		import matplotlib.pyplot as plt

		plt.plot(self, **kwargs)
		if not self.has_sublayers:
			plt.xlabel("Layer")

		else:
			plt.xlabel("Sublayer")
			plt.xticks(
				labels=[
				l for i, l in enumerate(self.labels()) if i % tick_spacing == 0
			],
			ticks = range(0, len(self), tick_spacing),
			rotation = 60,
			)

	def mean_update(self, other: "ResidualStream", i: int) -> "ResidualStream":
		"""Interpret `self` as a running mean and update with `other`.
		"""
		return self.zip_map(lambda mu, x: i / (i + 1) * mu + 1 / (i + 1) * x, other)

	def residuals(self) -> "ResidualStream":
		"""Compute residual (hidden state diff) for each block.
		"""
		return self.pairwise_map(lambda s1, s2: s2 - s1)

	def __contains__(self, item: torch.Tensor) -> bool:
		return any(item.is_set_to(state) for state in self)

	@overload
	def __getitem__(self, item: slice) -> list[torch.Tensor]:
		...

	@overload
	def __getitem__(self, item: int) -> torch.Tensor:
		...

	def __getitem__(self, item: Union[int, slice]):
		return list(self)[item]

	def __iter__(self) -> Generator[torch.Tensor, None, None]:
		for _, state in self.items():
			yield state

	def __len__(self) -> int:
		num_states = len(self.attentions) + len(self.layers)
		if self.embeddings is not None:
			num_states += 1 
		return num_states 

	def __reversed__(self) -> Generator[torch.Tensor, None, None]:
		for _, state in self.items(reverse = True):
			yield state

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
	module in the transformer, storing the input in a dictionary keyed by the 
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

		residual_stream.layers.append(process(output))

	_, layers = get_model_layers(model)

	#Since embedding layer is not in ModuleList, we register a prehook on the input
	if include_input:
		hooks.append(layers[0].register_forward_pre_hook(store_embeddings))

	for i, layer in enumerate(layers):
		# There should be a layernorm before and after attention. 
		# register hook on input to next layer
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
			# need to double check the logic is correct here.
			# Get the post attention layer norm.
			# In ViT, it is applied after the residual
			post_attention_ln = layer_norms[0 if post_norm else 1]
			hooks.append(post_attention_ln.register_forward_pre_hook(store_attention))

	try:
		yield residual_stream

	finally:
		for hook in hooks:
			hook.remove()








