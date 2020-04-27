import gpucommunity
import numpy as np
import numpy.testing as npt
import networkx as nx
import os
from scipy import io


def test_create():
    karate = nx.karate_club_graph()
    gpu_comm = gpucommunity.Community(karate)

def test_community():
    karate = nx.karate_club_graph()
    Q = karate
    #Q = nx.Graph()
    #Q.add_edge(0,1,weight=8)
    #Q.add_edge(2,1,weight=1)
    #Q.add_edge(3,1,weight=1)
     
    gpu_comm = gpucommunity.Community(Q)
    print(f"[T]Initial modularity {gpu_comm.modularity}")
    
    #do some mod
    gpu_comm.compute_one_level(limit=1)
    res = gpu_comm.partition
    print(f"[T]result {res}")
    print(f"[T]incidents {gpu_comm.communities_sum_incidents}")
    print(f"[T]inner weights {gpu_comm.communities_sum_inside}")
    
def test_merge():

    Q = nx.Graph()
    Q.add_edge(0,1,weight=8)
    Q.add_edge(2,1,weight=1)
    Q.add_edge(3,1,weight=1)
    
    gpu_comm = gpucommunity.Community(Q)
    print(f"[T]Initial modularity {gpu_comm.modularity}")
    
    #do some mod
    gpu_comm.compute_one_level(limit=6)
    res = gpu_comm.partition
    print(f"[T]result {res}")
    print(f"[T]incidents {gpu_comm.communities_sum_incidents}")
    print(f"[T]inner weights {gpu_comm.communities_sum_inside}")
    
    gpu_comm.merge_communities()
    res = gpu_comm.nxgraph.edges(data=True)
    print(f"[T]result {res}")

    
def test_compute_n_merge():

    Q = nx.Graph()
    Q.add_edge(0,1,weight=8)
    Q.add_edge(2,1,weight=1)
    Q.add_edge(3,1,weight=1)
    
    gpu_comm = gpucommunity.Community(Q)
    print(f"[T]Initial modularity {gpu_comm.modularity}")

    
    #do some mod
    gpu_comm.compute_one_level(limit=6)
    res = gpu_comm.partition
    print(f"[T]result {res}")
    print(f"[T]incidents {gpu_comm.communities_sum_incidents}")
    print(f"[T]inner weights {gpu_comm.communities_sum_inside}")
    
    gpu_comm.merge_communities()
    res = gpu_comm.nxgraph.edges(data=True)
    print(f"[T]com nxgraph edges {res}")
    print(f"[T]com incidents {gpu_comm.communities_sum_incidents}")
    print(f"[T]com inner weights {gpu_comm.communities_sum_inside}")
    
def test_n_iterations():
    Q = nx.karate_club_graph()
    
    #Q = nx.Graph()
    #Q.add_edge(0,1,weight=8)
    #Q.add_edge(2,1,weight=1)
    #Q.add_edge(3,1,weight=1)

    gpu_comm = gpucommunity.Community(Q)
    print(f"[T]Initial modularity {gpu_comm.modularity}")
    
    gpu_comm.compute_one_level(limit=4)
    res = gpu_comm.partition
    print(f"[T]partition after one level {res}")
    
    gpu_comm.merge_communities()
    gpu_comm.compute_one_level(limit=4)
    res = gpu_comm.partition
    print(f"[T]partition after 2nd level {res}")
    
    gpu_comm.merge_communities()
    gpu_comm.compute_one_level(limit=4)
    res = gpu_comm.partition
    print(f"[T]partition after 3nd level {res}")
    
    for _ in range(10):
        gpu_comm.merge_communities()
        gpu_comm.compute_one_level(limit=4)
        res = gpu_comm.partition.copy()
        print(f"[T]partition after {_} level {res}")
        
    print(f"[T]modularity is {gpu_comm.modularity}")
    
def test_best_partition1():
    __MIN = 0.001
    status_list = []
    Q = nx.karate_club_graph()

    gpu_comm = gpucommunity.Community(Q)
    old_mod = gpu_comm.modularity
    print(f"[T]Initial modularity {old_mod}")
    
    gpu_comm.compute_one_level(limit=2)
    new_mod = gpu_comm.modularity
    partition = gpu_comm.partition.copy()
    status_list.append(partition)
    old_mod = new_mod
    gpu_comm.merge_communities()
    loop = 0
    while True:
        loop +=1
        gpu_comm.compute_one_level(limit=2)
        new_mod = gpu_comm.modularity
        if new_mod - old_mod < __MIN:
            print('[T] BREAK!!!')
            break
        partition = gpu_comm.partition.copy()
        status_list.append(partition)
        old_mod = new_mod
        gpu_comm.merge_communities()
    
    print(f"levels {len(status_list)}")
    print(f"[T]{status_list[-1]}")
    print(f"[T]{set(status_list[-1].values())}")
    print(f"[T]{new_mod}")
    print(f'loops{loop}')
    
def test_best_partition2():
    MIN = 0.001

    #Q = nx.karate_club_graph()
    coo_mtx = io.mmread(os.path.join(os.getcwd(),'graphs/bio-diseasome.mtx'))
    Q = nx.from_scipy_sparse_matrix(coo_mtx)

    gpu_comm = gpucommunity.Community(Q)
    gpu_comm.compute_one_level(limit=6)
    res = gpu_comm.partition
    print(f"[T]result {res}")
    print(f"[T]incidents {gpu_comm.communities_sum_incidents}")
    print(f"[T]inner weights {gpu_comm.communities_sum_inside}")
    #status_list = gpu_comm.best_partition(limit=6, __MIN=MIN)
    mod = gpu_comm.modularity
