import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_sparse import matmul
from torch_geometric.typing import Adj, Size

class LCMessages(MessagePassing):
    def __init__(self, d):
        super(LCMessages, self).__init__(aggr=None)
        self.d = d
        self.C_sq = nn.LSTM(input_size=d*4, hidden_size=d, bias=True)
        self.C_equi = nn.LSTM(input_size=d*4, hidden_size=d, bias=True)
        self.C_diamond = nn.LSTM(input_size=d*4, hidden_size=d, bias=True)
        self.C_midpoint = nn.LSTM(input_size=d*4, hidden_size=d, bias=True)  # Keep d*4 for consistency

    def forward(self, x_p, x_c, x_c_h, c_t, p2c):
        # Extract indices for each constraint type
        equi_idx = torch.nonzero(c_t == 0).view(-1)
        squares_idx = torch.nonzero(c_t == 1).view(-1)
        diamond_idx = torch.nonzero(c_t == 2).view(-1)
        midpoint_idx = torch.nonzero(c_t == 3).view(-1)

        # Handle point gathering and padding for all constraints
        vars = x_p[p2c].view(-1, 4, x_p.shape[-1])  # Reshape to (num_constraints, 4, d)
        
        # Process each constraint type
        x_c_ = torch.empty_like(x_c).to(x_p.device)
        x_c_h_ = torch.empty_like(x_c_h).to(x_p.device)

        # Process squares
        if len(squares_idx) > 0:
            vars_squares = vars[squares_idx].view(-1, 4*self.d)
            _, (x_c_squares, x_c_h_squares) = self.C_sq(vars_squares.unsqueeze(0),
                                       (x_c[squares_idx].unsqueeze(0),
                                        x_c_h[squares_idx].unsqueeze(0)))
            x_c_[squares_idx] = x_c_squares.squeeze(0)
            x_c_h_[squares_idx] = x_c_h_squares.squeeze(0)

        # Process equisegments
        if len(equi_idx) > 0:
            vars_equi = vars[equi_idx].view(-1, 4*self.d)
            _, (x_c_equi, x_c_h_equi) = self.C_equi(vars_equi.unsqueeze(0),
                                       (x_c[equi_idx].unsqueeze(0),
                                        x_c_h[equi_idx].unsqueeze(0)))
            x_c_[equi_idx] = x_c_equi.squeeze(0)
            x_c_h_[equi_idx] = x_c_h_equi.squeeze(0)

        # Process diamonds
        if len(diamond_idx) > 0:
            vars_diamond = vars[diamond_idx].view(-1, 4*self.d)
            _, (x_c_diamond, x_c_h_diamond) = self.C_diamond(vars_diamond.unsqueeze(0),
                                           (x_c[diamond_idx].unsqueeze(0),
                                            x_c_h[diamond_idx].unsqueeze(0)))
            x_c_[diamond_idx] = x_c_diamond.squeeze(0)
            x_c_h_[diamond_idx] = x_c_h_diamond.squeeze(0)

        # Process midpoints (with padding)
        if len(midpoint_idx) > 0:
            vars_midpoint = vars[midpoint_idx]
            # Add zero padding for the 4th point
            padding = torch.zeros_like(vars_midpoint[:, 0]).unsqueeze(1)
            vars_midpoint = torch.cat([vars_midpoint[:, :3], padding], dim=1)
            vars_midpoint = vars_midpoint.view(-1, 4*self.d)
            
            _, (x_c_midpoint, x_c_h_midpoint) = self.C_midpoint(vars_midpoint.unsqueeze(0),
                                             (x_c[midpoint_idx].unsqueeze(0),
                                              x_c_h[midpoint_idx].unsqueeze(0)))
            x_c_[midpoint_idx] = x_c_midpoint.squeeze(0)
            x_c_h_[midpoint_idx] = x_c_h_midpoint.squeeze(0)

        return x_c_, x_c_h_

class CLMessages(MessagePassing):

    def __init__(self, d):
        super(CLMessages, self).__init__(aggr=None)
        self.d = d
        self.L_u = nn.LSTM(input_size=d,
                               hidden_size=d,
                               bias=True)

    def message_and_aggregate(self, adj, x_c):
        return matmul(adj.t(), x_c) # n_clauses x d

    def forward(self, adj, x_c, x_p, x_p_h, p_t, vars_idx,fixed_idx):
        msg = self.propagate(adj, x_c=x_c) # num_lits x d
        # get indices of ones in p_t
        ones_idx = vars_idx
        msg_ones = msg[ones_idx]
        x_p_ones = x_p[ones_idx]
        x_p_h_ones = x_p_h[ones_idx]

        _, (x_p_ones, x_p_h_ones) = self.L_u(msg_ones.unsqueeze(0),
                                   (x_p_ones.unsqueeze(0),#.detach(),
                                    x_p_h_ones.unsqueeze(0))#.detach())
                                  )
        x_p_ = torch.empty_like(x_p).to(x_p.device)
        x_p_h_ = torch.empty_like(x_p_h).to(x_p.device)
        x_p_[ones_idx] = x_p_ones.squeeze(0)
        x_p_[fixed_idx] = x_p[fixed_idx]
        x_p_h_[ones_idx] = x_p_h_ones.squeeze(0)
        x_p_h_[fixed_idx] = x_p_h[fixed_idx]
        return x_p_, x_p_h_

class NeuroSAT(nn.Module):

    def __init__(self, d,
                 n_msg_layers=0,
                 n_vote_layers=0,
                 mlp_transfer_fn = 'relu',
                 final_reducer = 'mean',
                 lstm = 'standard',
                 return_embs = False,
                 max_size=20,
                ):
        super(NeuroSAT, self).__init__()

        self.d = d
        self.return_embs = return_embs
        self.final_reducer = final_reducer
        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, d)
        self.C_init = nn.Linear(1, d)

        self.LC_msgs = LCMessages(d=d)
        self.CL_msgs = CLMessages(d=d)
        vocab_size = max_size ** 2
        self.embedding = nn.Embedding(vocab_size, d)
        self.classifier = nn.Linear(d, vocab_size)
        self.classifier.weight = self.embedding.weight

    def forward(self, data, num_iters):
        adj_t = data.adj_t
        n_lits, n_clauses = data.x_p.shape[0], data.x_c.shape[0]
        c_t = data.c_type.to(data.x_p.device)
        p_t = data.p_type.to(data.x_p.device)
        y = data.y.to(data.x_p.device)
        embs = self.embedding(y)
        
        # Get indices of known and unknown points
        fixed_p_idx = torch.nonzero(p_t).view(-1)
        vars_idx = torch.nonzero(1-p_t).view(-1)
        
        embs_ones = embs[fixed_p_idx]
        p2c = data.constr_points_ixs.to(data.x_p.device)
        
        # Initialize embeddings
        init_ts = self.init_ts.to(data.x_p.device)
        x_p_ = torch.rand((n_lits, self.d), requires_grad=True).to(data.x_p.device)
        x_p = torch.empty_like(x_p_).to(data.x_p.device)
        x_p[fixed_p_idx] = embs_ones
        x_p[vars_idx] = x_p_[vars_idx]
        
        C_init = self.C_init(init_ts)
        x_c = C_init.repeat(n_clauses, 1)

        x_p_batch = data.x_p_batch
        x_p_h = torch.zeros(x_p.shape).to(data.x_p.device)
        x_c_h = torch.zeros(x_c.shape).to(data.x_p.device)
        
        intermediate_results = [[x_p, self.classifier(x_p)]]

        for t in range(num_iters):
            x_c_, x_c_h = self.LC_msgs(x_p, x_c, x_c_h, c_t, p2c)
            x_p, x_p_h = self.CL_msgs(adj_t, x_c_, x_p, x_p_h, p_t, vars_idx, fixed_p_idx)
            x_c = x_c_
            intermediate_results.append([x_p, self.classifier(x_p)])
        
        logits = self.classifier(x_p)
        return logits, intermediate_results

