from typing import Optional

import torch

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils import dtype_from_name


class SaveForOutput(torch.nn.Module, GraphModuleMixin):
    """Copy a field and disconnect it from the autograd graph.

    Copy a field and disconnect it from the autograd graph, storing it under another key for inspection as part of the models output.

    Args:
        field: the field to save
        out_field: the key to put the saved copy in
    """

    field: str
    out_field: str

    def __init__(self, field: str, out_field: str, irreps_in=None):
        super().__init__()
        self._init_irreps(irreps_in=irreps_in)
        self.irreps_out[out_field] = self.irreps_in[field]
        self.field = field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = data[self.field].detach().clone()
        return data


class ComputeDtypeWrapper(torch.nn.Module, GraphModuleMixin):
    """Wrap a model, converting all floating point inputs to ``compute_dtype`` and all outputs of ``compute_dtype`` back to ``output_dtype``"""

    compute_dtype: torch.dtype
    output_dtype: torch.dtype

    def __init__(
        self,
        func: GraphModuleMixin,
        compute_dtype: torch.dtype,
        output_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.compute_dtype = dtype_from_name(compute_dtype)
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        self.output_dtype = dtype_from_name(output_dtype)
        self.func = func
        self._init_irreps(irreps_in=func.irreps_in, irreps_out=func.irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # convert the input
        inp = {}
        for k, v in data.items():
            if v.is_floating_point():
                inp[k] = v.to(dtype=self.compute_dtype)
            else:
                inp[k] = v
        # run the model
        data = self.func(inp)
        # convert the output
        # TODO: need to think about how not to waste compute here...
        out = {}
        for k, v in data.items():
            if v.dtype == self.compute_dtype:
                out[k] = v.to(dtype=self.output_dtype)
            else:
                out[k] = v
        return out
