import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.dataset import files_exist, __repr__
from torch_sparse import SparseTensor
import pytorch_lightning as pl
from os import listdir
import os.path as osp
import pickle as pickle
from collections import defaultdict
#import PyMiniSolvers.minisolvers as minisolvers

__all__ = ['solve_sat']


class Problem(Data):
    # a Problem is a bipartite graph

    def __init__(self,x_p, x_c, edge_index, p_type, c_type, y, constr_points_ics):
            #self, edge_index=None, x_l=None, x_c=None, y=None):
        # edge_index is a bipartite adjacency matrix between lits and clauses
        # x_l is the feature vector of the lits nodes
        # x_c is the feature vector of the clauses nodes
        # y is the label: 1 if sat, 0 if unsat

        super(Problem, self).__init__()
        # nodes features
        self.x_p = x_p
        self.x_c = x_c
        self.y = y
        # create self.z so that is has same dimension as x_l and random binary values
        self.p_type = p_type if p_type is not None else 0
        self.c_type = c_type if c_type is not None else 0

        self.num_vars = x_p.size(0) if x_p is not None else 0
        self.num_clauses = x_c.size(0) if x_c is not None else 0
        self.constr_points_ixs = constr_points_ics
        self.check_cpi = constr_points_ics
        # edges
        self.edge_index = edge_index
        self.adj_t = SparseTensor(row = edge_index[0],
                                  col = edge_index[1],
                                  sparse_sizes = [self.num_clauses, self.num_vars]
                                 ) if edge_index is not None else 0

        # compute number of variables
        #assert self.num_literals %2 == 0
        self.num_nodes = self.num_vars + self.num_clauses

    def __inc__(self, key, value,store):
        if key == 'constr_points_ixs':
            return self.num_vars
        if key == 'edge_index':
            return torch.tensor([[self.x_p.size(0)], [self.x_c.size(0)]])
        else:
            return super().__inc__(key, value)


# create pytorch geometric dataset

# Remark: using a InMemoryDataset is faster than a standard Dataset

def map_point_to_integer(point, max_size):
    x, y = point
    return x + y * max_size

def preprocess_problem(problem, max_size):
    points = problem['points']
    points_order = list(points.keys())
    known_points = problem["new_points"]
    points_mark = [1 if pnt in known_points else 0 for pnt in points_order]
    constrs = problem['constrs']
    constr_clean = [const for const in constrs if const[0] != "segment"]  
    
    constraint_type_to_index = {
        "PARALLEL": 0,
        "SQUARE": 1,
        "MIDPOINT": 3,
        "DIAMOND": 2
    }
    
    constr_nodes = []
    constrs_mark = []
    
    for const in constr_clean:
        constr_type = const[0].upper()
        if constr_type == "MIDPOINT":
            # For midpoint constraint, pad with the first point to make it 4 points
            points_in_constraint = const[1:4]  # Get the 3 points
            padded_points = list(points_in_constraint) + [points_in_constraint[0]]  # Add first point as padding
            constr_nodes.append(tuple(padded_points))  # Convert to tuple for dictionary key
        else:
            # Other constraints already have 4 points
            constr_nodes.append(tuple(const[1:]))
            
        constrs_mark.append(constraint_type_to_index[constr_type])
    
    point_nodes = list(points.keys())
    grapf_index_dict_points = {points_order[i]: i for i in range(len(points_order))}
    grapf_index_dict_constrs = {constr_nodes[i]: i for i in range(len(constr_nodes))}
    
    point_to_constr_edges = [
        [grapf_index_dict_points[points_order[i]], grapf_index_dict_constrs[constr_nodes[j]]]
        for i in range(len(points_order))
        for j in range(len(constr_nodes))
        if points_order[i] in constr_nodes[j]
    ]
    
    constr_to_point_edges = [[a[1], a[0]] for a in point_to_constr_edges]
    
    y = []
    for p in points_order:
        y.append(map_point_to_integer(points[p], max_size))
    
    constraint_points_ordered = []
    for c in constr_nodes:
        for p in c:
            constraint_points_ordered.append(grapf_index_dict_points[p])
            
    return constr_to_point_edges, y, points_mark, constrs_mark, constraint_points_ordered


def make_pyg_datapoint(problem,max_size,d):
  
  constr_to_point_edges, y, points_mark, constrs_mark, constraint_points_ordered = preprocess_problem(problem,max_size)
  
  lab = torch.tensor(y, dtype=torch.long)
  x_p = torch.randn((len(points_mark), d))
  x_c = torch.randn((len(constrs_mark), d))

  p_type = torch.tensor(points_mark, dtype=torch.long)
  c_type = torch.tensor(constrs_mark, dtype=torch.long)

  edge_index = torch.tensor(constr_to_point_edges, dtype=torch.long).t().contiguous()

  constr_points_ics = torch.tensor(constraint_points_ordered, dtype=torch.long)

  return Problem(x_p = x_p, x_c = x_c, edge_index=edge_index, p_type = p_type, c_type = c_type, y = lab, constr_points_ics=constr_points_ics)


def get_Geo_data(data, d, max_size):
    dataset = []
    for i in range(len(data)):
        problem = make_pyg_datapoint(data[i],max_size,d)
        dataset.append(problem)
    return dataset


def get_Geo_dataset(dataset_file, max_size, max_constraints, valid_split):
    with open(dataset_file, 'rb') as f:
        problems = pickle.load(f)
    data = []
    for pr in problems:
        num_constr = len([c for c in pr['constrs'] if c[0]!='segment'])
        maxx = max([x for x,y in pr['points'].values()])
        maxy = max([y for x,y in pr['points'].values()])
        if max(maxx,maxy) < max_size and num_constr < max_constraints:
            data.append(pr)
    data = data
    trainlen = int(len(data)*valid_split)
    
    train_data = data[:trainlen]
    val_data = data[trainlen:]
    d = 128
    dataset_train = get_Geo_data(train_data, d, max_size)
    dataset_val = get_Geo_data(val_data, d, max_size)
    #dataset_test = InMemoryGeoDataset(dimacs_dir_tes, d)
    return [dataset_train, dataset_val, dataset_val]


class Geo_datamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super(Geo_datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.follow_batch = ['x_p','x_c']

    def setup(self, stage=None):
        self.train_dataset = self.dataset[0]
        self.val_dataset = self.dataset[1]
        self.test_dataset = self.dataset[2]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,  follow_batch=self.follow_batch, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, follow_batch=self.follow_batch, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  follow_batch=self.follow_batch, batch_size=self.batch_size)

    def group_data(dataset):
        # create a dictionary of train sets based on the number of literals
        train_sets = defaultdict(list)
        for pr in dataset:
            num_lits = pr.x_l.shape[0]
            train_sets[num_lits].append(pr)
        return train_sets