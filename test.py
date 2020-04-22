import gpucommunity
import numpy as np
import numpy.testing as npt
import networkx as nx
  
def test_community():
    dummy_arr = np.array([1,2,2,2], dtype=np.int32)
    karate = nx.karate_club_graph()

    Q = nx.Graph()
    Q.add_edge(0,1,weight=1)
    Q.add_edge(2,1,weight=1)
    Q.add_edge(3,1,weight=10)
    
    
    #gpu_comm = gpucommunity.GPUCommunity(dummy_arr, karate)
    #gpu_comm = gpucommunity.GPUCommunity.__new__(gpucommunity.GPUCommunity, dummy_arr, karate) 
    gpu_comm = gpucommunity.Community(dummy_arr, Q)
    print(f"[T]Initial modularity {gpu_comm.modularity}")
    gpu_comm.gpu.increment()
    gpu_comm.gpu.retreive_inplace()
    want = [2,3,3,3]
    npt.assert_array_equal(dummy_arr,want)
    
    gpu_comm.compute_one_level(limit=4)
    res = gpu_comm.partition
    print(f"results = {res}")
    

    
