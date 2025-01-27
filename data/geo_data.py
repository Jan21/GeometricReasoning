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

class InMemoryGeoDataset(InMemoryDataset):
    def __init__(self, root, d):
        # root: location of dataset
        # d: number of features for x_l and x_c

        self.root = root
        self.d = d

        # create initial feature vectors
        self.l_init = torch.normal(mean=0.0, std=1.0, size=(1,self.d)) # original
        self.c_init = torch.normal(mean=0.0, std=1.0, size=(1,self.d)) # original
        #self.l_init = torch.zeros(size=(1,self.d), dtype=torch.float32)
        #self.c_init = torch.zeros(size=(1,self.d), dtype=torch.float32)
        self.denom = torch.sqrt(torch.tensor(self.d, dtype=torch.float32)) # or float64???

        super(InMemoryGeoDataset, self).__init__(root=self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        # get dimacs filenames
        sorted_file_names = sorted([f for f in listdir(self.root)
                                   if osp.isfile(osp.join(self.root,f))])
        return sorted_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']


    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        for raw_path in self.raw_paths:
            n_vars, clauses = self.parse_dimacs(raw_path)
            # n_vars is the number of variables according to the dimacs file
            # clauses is a list of lists (=clauses) of numbers (=literals)

            y, _ = solve_sat(n_vars, clauses) # get problem label (sat/unsat)

            # create graph instance (Problem)
            p = self.create_problem(n_vars, clauses, y)

            data_list.append(p)
        #for i in range(10):
        #    with open(self.root+f"/processed{str(i)}.pkl",'wb') as f:
        #        pickle.dump(data_list[20000*i:20000*i+20000], f)
        #with open(self.root+"/processed2.pkl",'wb') as f:
        #    pickle.dump(data_list[100000:], f)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # parse dimacs file
    def parse_dimacs(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0
        while lines[i].strip().split(" ")[0] == "c":
            # strip : remove spaces at the beginning and at the end of the string
            i += 1

        header = lines[i].strip().split(" ")
        assert(header[0] == 'p')
        n_vars = int(header[2])
        clauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i+1:]]
        return n_vars, clauses

    # create Problem instance (graph) from parsed dimacs
    def create_problem(self, n_vars, clauses, y):
        # d is the number of features of x_l and x_c

        n_lits = int(2 * n_vars)
        n_clauses = len(clauses)

        # create feature vectors for lits and clauses
        x_l = (torch.div(self.l_init, self.denom)).repeat(n_lits, 1)
        x_c = (torch.div(self.c_init, self.denom)).repeat(n_clauses, 1)

        # get graph edges from list of clauses
        edge_index = [[],[]]
        for i,clause in enumerate(clauses):
            # get idxs of lits in clause
            lits_indices = [self.from_lit_to_idx(l, n_vars) for l in clause]
            clauses_indices = len(clause) * [i]

            # add all edges connected to clause i to edge_index
            edge_index[0].extend(lits_indices)
            edge_index[1].extend(clauses_indices)

        # convert edge_index to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        return Problem(edge_index, x_l, x_c, y)

    def from_lit_to_idx(self, lit, n_vars):
        # from a literal in range {1,...n_vars,-1,...,-n_vars} get the literal
        # index in {0,...,n_lits-1} = {0,...,2*n_vars-1}
        # if l is positive l <- l-1
        # if l in negative l <- n_vars-l-1
        assert(lit!=0)
        if lit > 0 :
            return lit - 1
        if lit < 0 :
            return n_vars - lit - 1

    def from_index_to_lit(self, idx, n_vars):
        # inverse of 'from_lit_to_idx', just in case
        if idx < n_vars:
            return idx+1
        else:
            return n_vars-idx-1



def map_point_to_integer(point, max_size):
    x, y = point
    return x + y * max_size

def preprocess_problem(problem,max_size):
  points = problem['points']
  points_order = list(points.keys())
  known_points = problem["new_points"]
  points_mark = [1 if pnt in known_points else 0 for pnt in points_order]

  constrs = problem['constrs']
  constr_clean = [const for const in constrs if const[0] != "segment"] #order
  constr_nodes = [const[1:] for const in constr_clean] #order
  constrs_mark = [0 if cnst == "equisegment" else 1 for cnst in [c[0] for c in constr_clean]]

  point_nodes = list(points.keys())

  grapf_index_dict_points = {points_order[i] : i for i in range(len(points_order))}
  grapf_index_dict_constrs = {constr_nodes[i] : i for i in range(len(constr_nodes))}

  point_to_constr_edges = [[grapf_index_dict_points[points_order[i]], grapf_index_dict_constrs[constr_nodes[j]]] for i in range(len(points_order)) for j in range(len(constr_nodes)) if points_order[i] in constr_nodes[j]]
  constr_to_point_edges = [[a[1], a[0]] for a in point_to_constr_edges]

  y = []
  for p in points_order:
    y.append(map_point_to_integer(points[p],max_size))

  
  constraint_points_ordered = []
  for c in constr_nodes:
    for p in c:
        constraint_points_ordered.append(grapf_index_dict_points[p])

  return constr_to_point_edges, y, points_mark, constrs_mark, constraint_points_ordered

def make_pyg_datapoint(problem,max_size):
  
  constr_to_point_edges, y, points_mark, constrs_mark, constraint_points_ordered = preprocess_problem(problem,max_size)
  
  lab = torch.tensor(y, dtype=torch.long)
  x_p = torch.randn((len(points_mark), 128))
  x_c = torch.randn((len(constrs_mark), 128))

  p_type = torch.tensor(points_mark, dtype=torch.long)
  c_type = torch.tensor(constrs_mark, dtype=torch.long)

  edge_index = torch.tensor(constr_to_point_edges, dtype=torch.long).t().contiguous()

  constr_points_ics = torch.tensor(constraint_points_ordered, dtype=torch.long)

  return Problem(x_p = x_p, x_c = x_c, edge_index=edge_index, p_type = p_type, c_type = c_type, y = lab, constr_points_ics=constr_points_ics)


def get_Geo_data(data, d, max_size):
    dataset = []
    for i in range(len(data)):
        problem = make_pyg_datapoint(data[i],max_size)
        dataset.append(problem)
    return dataset


def get_Geo_dataset(dataset_file):
    with open(dataset_file, 'rb') as f:
        problems = pickle.load(f)
    data = []
    max_size = 30 # TADY bylo 10
    for pr in problems:
        num_constr = len([c for c in pr['constrs'] if c[0]!='segment'])
        maxx = max([x for x,y in pr['points'].values()])
        maxy = max([y for x,y in pr['points'].values()])
        if max(maxx,maxy) < max_size and num_constr < 20:  # TADY bylo 4  20
            data.append(pr)
    data = data
    trainlen = int(len(data)*0.9)
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