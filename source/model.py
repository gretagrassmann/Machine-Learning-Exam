import torch
import torch.nn as nn
from torch.nn import Linear, GRU, Parameter
import torch.nn.functional as F
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_



class MultiHeadTripletAttention(MessagePassing):
    """
    Pytorch Geometric provides the MessagePassing base class, which helps in creating message passing GNNs by
    automatically taking care of message propagation. The user only has to define the message passing
    function (-> def message()), the update (-> def update()) and the aggregation scheme (-> aggr='add'). node_dim is an
    attribute that indicates along which axis to propagate.
    """
    def __init__(self, node_channels, edge_channels, heads=3, negative_slope=0.2, **kwargs):
        super(MultiHeadTripletAttention, self).__init__(aggr='add', node_dim=0, **kwargs) #!!!aggr='mean'
        # node_dim = 0 for multi-head aggr support
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()


    def reset_parameters(self):
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        self.propagate() is the initial call to start propagating messages. It takes in the edge indices and all
        additional data which is needed to construct messages and to update node embeddings.
        if (edge_index, edge_attr) represent the matrix M in the sense that M is the adjacency matrix where the entries
        are populated using the edge attribute held in parameter weight (when an edge does not have that attribute, the
        value of the entry is 1, for multiple edges the matrix values are the sums of the edge weights) then
        .propagate(edge_index, x, edge_attr) computes M_transpose @ x.
        """
        # Initial x=[#,head*node_channels]
        x = torch.matmul(x, self.weight_node) #matmlu=matrix product
        # Final x=[#nodes,node_channels]
        # Initial edge_attr=[#,#edge_features]
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        # Final edge_attr = [#,head*node_channels]
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size) # [#,node_channels]

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        """
        Constructs messages to each node from each edge. Can take any argument which was initially passed to
        self.propagate() (used in def forward()) . Tensor passed to self.propagate() can be mapped to the respective
        nodes i and j by appending _i or _j to the variable names (i.e. x_i and x_j).
        """
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.node_channels)  #[#,heads,node_channels]
        x_i = x_i.view(-1, self.heads, self.node_channels)   #[#,heads,node_channels]
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)  #[#,heads,node_channels]

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)  # time consuming 13s [#,heads,3*node_channels]
        """torch.cat() concatenates the given sequence in the given dimension. All tensor must have the same shape, 
        except in the concatenating dimension."""
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)  # time consuming 12.14s
        # Before the sum [#,heads,3*node_channels]. After the sum [#, heads]
        alpha = leaky_relu(alpha, self.negative_slope)  # [#, heads]
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i) # [#, heads]
        alpha = alpha.view(-1, self.heads, 1)  # [#, heads]
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        """
        Updates node embeddings . Takes in the output of aggregation as first argument and any argument which was
        initially passed to self.propagate().
        """
        # Initial aggr_out = [#, heads, node_channels]
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        # Final aggr_out = [#, heads*node_channels]
        aggr_out = torch.matmul(aggr_out, self.weight_scale) #[#,node_channel]
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self):
        return '{node_channels}, {node_channels}, heads={heads}'.format(**self.__dict__)


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, heads=4, time_step=3):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        #x = [#,node_channel]
        h = x.unsqueeze(0) # [1,#,node_channel]
        for i in range(self.time_step):
            # conv <- MultiHeadTripletAttention [#, node_channels]
            m = F.celu(self.conv.forward(x, edge_index, edge_attr)) # [#, node_channels]
            x, h = self.gru(m.unsqueeze(0), h) # both [1,#,node_channels]
            """
            m (seq_len, batch, input_size): tensor containing the features of the input sequence. 
            h (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for 
            each element in the batch. Defaults to zero if not provided. 
            If the RNN is bidirectional, num_directions should be 2, else it should be 1.
            """
            x = self.ln(x.squeeze(0)) # [#, node_channels]
        return x


class TrimNet(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=32, depth=3, heads=4, dropout=0.1, outdim=2):
        super(TrimNet, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.lin0 = Linear(in_dim, hidden_dim) #in_dim=number of node features, hidden dim= number of node channels
        self.convs = nn.ModuleList([
            Block(hidden_dim, edge_in_dim, heads)
            for i in range(depth)
        ]) # ModuleList contains depth(=3) Block
        self.set2set = Set2Set(hidden_dim, processing_steps=3)

        """
        def __init__(self, in_channels, processing_steps, num_layers=1):
            super(Set2Set, self).__init__()

            self.in_channels = in_channels   #=hidden_dim
            self.out_channels = 2 * in_channels
            self.processing_steps = processing_steps
            self.num_layers = num_layers
    
            self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,  #in_channels=hiddend_dim of Set2set
                                      num_layers)
    
            self.reset_parameters()

        def reset_parameters(self):
        self.lstm.reset_parameters()


       def forward(self, x, batch):
            batch_size = batch.max().item() + 1     #=128
    
            h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),  #Returns a Tensor filled with 0.
                 x.new_zeros((self.num_layers, batch_size, self.in_channels)))
            q_star = x.new_zeros(batch_size, self.out_channels)
    
            for i in range(self.processing_steps):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.in_channels)
                e = (x * q[batch]).sum(dim=-1, keepdim=True)
                a = softmax(e, batch, num_nodes=batch_size)
                r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
                # scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0): Sums all values from the src
                # tensor into out at the indices specified in the index tensor along a given axis dim. For each value
                # in src, its output index is specified by its index in input for dimensions outside of dim and by the
                # corresponding value in index for dimension dim. If multiple indices reference the same location, 
                # their contributions add.
                q_star = torch.cat([q, r], dim=-1)
    
            return q_star

        """
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, outdim)
        )

    def forward(self, data):
        """
        data.x is the content of each considered batch [#atmos in the batch, node features]
        """
        x = F.celu(self.lin0(data.x)) # data.x [#,#node features] -lin0-> [#, hidden_dim]
        for conv in self.convs:
            # conv <- Block [#, hidden_dim]
            x = x + F.dropout(conv(x, data.edge_index, data.edge_attr), p=self.dropout, training=self.training)
        x = self.set2set(x, data.batch) # [batch_size, 2*hidden_dim]
        """ data.batch marks the atoms that belongs to each one of the 128 molecules of a batch"""
        x = self.out(F.dropout(x, p=self.dropout, training=self.training)) #[batch_size, 2]

        return x
