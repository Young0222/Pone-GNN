from typing import Optional
from typing import Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.typing import PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import degree
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


class LightGConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, inputs: Tensor) -> Tensor:
        return inputs


class LightGINConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.eps = torch.nn.Parameter(torch.empty(1))
        self.eps.data.fill_(0.0)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)


        norm_self = deg_inv_sqrt[range(x.size(0))] * deg_inv_sqrt[range(x.size(0))]
        norm_self = norm_self.unsqueeze(dim=1).repeat(1, x.size(1))
        out = out + (1 + self.eps) * norm_self * x  # GIN

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, inputs: Tensor) -> Tensor:
        return inputs


class LightGINConv2(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.first_aggr = first_aggr
        self.reset_parameters()
        self.eps = torch.nn.Parameter(torch.empty(1))
        self.eps.data.fill_(0.0)

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj):
        def get_norm(node, edge_index):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt
        def gin_norm(out, input_x, input_deg_inv_sqrt):
            norm_self = input_deg_inv_sqrt[range(input_x.size(0))] * input_deg_inv_sqrt[range(input_x.size(0))]
            norm_self = norm_self.unsqueeze(dim=1).repeat(1, input_x.size(1))
            out = out + (1 + self.eps) * norm_self * input_x  # GIN
            return out

        norm_pos, deg_inv_sqrt_pos = get_norm(x[0], pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(x[1], neg_edge_index)

        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x[0], size=None, norm=norm_pos)
            out_neg = self.propagate(neg_edge_index, x=x[0], size=None, norm=norm_neg)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[0], deg_inv_sqrt_neg)
            return out_pos, out_neg
        else:
            out_pos = self.propagate(pos_edge_index, size=None, x=x[0], norm=norm_pos)
            out_neg = self.propagate(pos_edge_index, size=None, x=x[1], norm=norm_pos)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[1], deg_inv_sqrt_pos)
            return out_pos, out_neg

    def message(self, x_j: Tensor, norm) -> Tensor:
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class LightSignedConv2(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj):

        # if isinstance(x, Tensor):
        #     x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor)

        if self.first_aggr:

            out_pos = self.propagate(pos_edge_index, x=x[0], size=None)
            out_pos = out_pos  # + x[0]

            out_neg = self.propagate(neg_edge_index, x=x[1], size=None)
            out_neg = out_neg  # + x[1]

            return out_pos, out_neg

        else:
            F_in = self.in_channels

            # positive features and pos edges
            out_pos = self.propagate(pos_edge_index, size=None, x=x[0])
            out_pos = out_pos  # + x[0]

            # positive node features and neg edges
            out_neg1 = self.propagate(pos_edge_index, size=None, x=x[1])
            out_neg = out_neg1  # + x[1]

            return out_pos, out_neg

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')


class LightSignedConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.first_aggr = first_aggr
        self.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj):
        def get_norm(node, edge_index):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm
        norm_pos = get_norm(x[0], pos_edge_index)
        norm_neg = get_norm(x[1], neg_edge_index)
        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x[0], size=None, norm=norm_pos)
            out_pos = out_pos
            out_neg = self.propagate(neg_edge_index, x=x[0], size=None, norm=norm_neg)
            out_neg = out_neg
            return out_pos, out_neg
        else:
            out_pos = self.propagate(pos_edge_index, size=None, x=x[0], norm=norm_pos)
            out_neg = self.propagate(pos_edge_index, size=None, x=x[1], norm=norm_pos)
            return out_pos, out_neg

    def message(self, x_j: Tensor, norm) -> Tensor:
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class SignedConv(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
        \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

    if :obj:`first_aggr` is set to :obj:`True`, and

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})},
        \mathbf{x}_v^{(\textrm{pos})} \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})},
        \mathbf{x}_v^{(\textrm{neg})} \right]

    otherwise.
    In case :obj:`first_aggr` is :obj:`False`, the layer expects :obj:`x` to be
    a tensor where :obj:`x[:, :in_channels]` denotes the positive node features
    :math:`\mathbf{X}^{(\textrm{pos})}` and :obj:`x[:, in_channels:]` denotes
    the negative node features :math:`\mathbf{X}^{(\textrm{neg})}`.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
          if bipartite,
          positive edge indices :math:`(2, |\mathcal{E}^{(+)}|)`,
          negative edge indices :math:`(2, |\mathcal{E}^{(-)}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos_l = Linear(in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
        else:
            self.lin_pos_l = Linear(2 * in_channels, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(2 * in_channels, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()

    def forward(
            self,
            x: Union[Tensor, PairTensor],
            pos_edge_index: Adj,
            neg_edge_index: Adj,
    ):

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: PairTensor)
        if self.first_aggr:

            out_pos = self.propagate(pos_edge_index, x=x)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1])

            out_neg = self.propagate(neg_edge_index, x=x)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1])

            return torch.cat([out_pos, out_neg], dim=-1)

        else:
            F_in = self.in_channels

            out_pos1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_pos2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos = out_pos + self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(pos_edge_index,
                                      x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_neg2 = self.propagate(neg_edge_index,
                                      x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg = out_neg + self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr})')


class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{e_{j,i}}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j

    Args:
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized via symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """

    def __init__(self, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize and isinstance(edge_index, Tensor):
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow,
                                  dtype=x.dtype)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
        improved: bool = False,
        add_self_loops: bool = True,
        flow: str = "source_to_target",
        dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
        \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. By default, self-loops will be added
            in case :obj:`normalize` is set to :obj:`True`, and not added
            otherwise. (default: :obj:`None`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on-the-fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: Optional[bool] = None,
            normalize: bool = True,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
