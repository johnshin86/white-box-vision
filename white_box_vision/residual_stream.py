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
	

	)

