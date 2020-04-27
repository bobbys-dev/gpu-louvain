import networkx as nx
import community
from scipy import io
import os
import numpy as np
import gpucommunity

def main():
    coo_mtx = io.mmread(os.path.join(os.getcwd(),'graphs/bio-diseasome.mtx')) 
    G = nx.from_scipy_sparse_matrix(coo_mtx)
    MIN = 0.001
    gpu_comm = gpucommunity.Community(G)
    gpu_comm.compute_one_level(limit=6)

    
if __name__ == "__main__":
    main()