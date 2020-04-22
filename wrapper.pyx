import numpy as np
import networkx as nx
cimport numpy as np

# assert sizeof(int) == sizeof(np.int32_t) 

# Using cython to bring in class constructor and attributes
cdef extern from "src/gpucommunity.hpp":
    cdef cppclass C_GPUCommunity "GPUCommunity":
        C_GPUCommunity(np.int32_t*, int, int, np.int32_t*, np.int32_t*, np.int32_t*, np.int32_t*, int, int, np.int32_t*, np.int32_t*, np.int32_t*, int)
        void setNodeCommunities(np.int32_t*)
        void increment()
        void retreive()
        void retreive_to(np.int32_t*, int)
        void getMaxDelta(double*, int*, int*)

cdef class GPUCommunity:
    cdef C_GPUCommunity* community_state 
    cdef int dummy_arr_dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] dummy_arr,
        np.ndarray[ndim=1, dtype=np.int32_t] node_degrees,
        np.ndarray[ndim=1, dtype=np.int32_t] node_communities,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_incidents,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_inside,
        nxgraph):
        """
        This takes in arrays of data about a graph to hold and update community data about the graph.
        
        dummy_arr: just pass some trivial array
        graph: networkx undirected graph
        
        csr: 3 arrays of compressed sparse row adjacency list
        node_degrees: 1D array sum of link weights incident to each node 
        communities_sum_incidents: 1D array sum of link weights incident to each community
        communities_sum_inside: 1D array sum of links inside each community
        
        To be passed to this C++ class constructor
        GPUCommunity::GPUCommunity (int* array_host_, int length_,
            const int nodes_count, 
            int node_degrees[],
            int communities_sum_incidents[],
            int communities_sum_inside[],
            const int nnz,
            const int csr_num_rows,
            int csr_data[],
            int csr_indices[],
            int csr_indptr[],
            int sum_all_weights)
        """
               
        # use local variables to ferry into constructor because its easier to read
        # Note creationg of cdef ndarray buffers are allowed only locally, not as member vars
        self.dummy_arr_dim1 = len(dummy_arr) # TODO: remove
        print("[~]Wow GPUCommunity")        
        
        # extract adj matrix in compressed sparse row
        csr = nx.to_scipy_sparse_matrix(nxgraph)
        csr_num_rows = len(csr.indptr) - 1
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_data = np.array(csr.data, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indices = np.array(csr.indices, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indptr = np.array(csr.indptr, dtype=np.int32)
        sum_all_weights = sum(csr.data)/2 # assumes undirected graph (and symmetric adj matrix
        
        # initialize community state in c/cuda
        self.community_state = new C_GPUCommunity(&dummy_arr[0],
            self.dummy_arr_dim1,
            len(node_degrees),
            &node_communities[0],
            &node_degrees[0],
            &communities_sum_incidents[0],
            &communities_sum_inside[0],
            csr.nnz,
            csr_num_rows,
            &csr_data[0],
            &csr_indices[0],
            &csr_indptr[0],
            sum_all_weights)
       

    def increment(self):
        self.community_state.increment()

    def retreive_inplace(self):
        self.community_state.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dummy_arr_dim1, dtype=np.int32)
        self.community_state.retreive_to(&a[0], self.dummy_arr_dim1)

        return a
        
    def get_max_delta(self):
        """Parallel computation
        """
        cdef double max
        cdef int node
        cdef int community
        self.community_state.getMaxDelta(&max, &node, &community)
        return [max, node, community]
    
    def set_node_communities(self, np.ndarray[ndim=1, dtype=np.int32_t] nc):
        """Sets array inplace
        """
        self.community_state.setNodeCommunities(&nc[0])    
            
class Community():
    def __init__(self, dummy_arr, nxgraph):
        print("[~]Whoah Community")

        self.node_degrees = np.array([d for n,d in nxgraph.degree(nxgraph.nodes(), weight='weight')], dtype=np.int32) 
        
        # init all nodes to their own singular communities
        # communities are labeled sequentially
        self.node_communities = np.array(nxgraph.nodes(), dtype=np.int32)

        # initial sum of incident links to communities is same as node's degrees
        # if no attr named 'weight' then defaults to link existence (0 or 1)
        self.communities_sum_incidents = np.array(self.node_degrees, dtype=np.int32) 
      
        # initial sum of internal links is 0
        self.communities_sum_inside = np.zeros(len(nxgraph), dtype=np.int32)
        
        self.nxgraph = nxgraph
        self.community_graph = nx.Graph()
        
        self.gpu = GPUCommunity(dummy_arr,
            self.node_degrees,
            self.node_communities,
            self.communities_sum_incidents,
            self.communities_sum_inside, #TODO: don't need this to calc delta
            self.nxgraph)
            
        #print(self.node_communities) #TODO: remove, it worked
        #self.node_communities[9] = 911 #TODO: removie
        #self.gpu.set_node_communities(self.node_communities) #TODO: removie
        #print(self.node_communities)#TODO: removie
        
        self.partition = dict(enumerate(self.node_communities))
         
        self.modularity = self.compute_modularity(self.partition, self.nxgraph)
    
    def compute_modularity(self, partition, nxgraph, weight='weight'):
        """Compute the modularity of a partition of a graph
        Parameters
        ----------
        partition : dict
           the partition of the nodes, i.e a dictionary where keys are their nodes
           and values the communities
        graph : networkx.Graph
           the networkx graph which is decomposed
        weight : str, optional
            the key in graph to use as weight. Default to 'weight'
        Returns
        -------
        modularity : float
           The modularity
        Raises
        ------
        KeyError
           If the partition is not a partition of all graph nodes
        ValueError
            If the graph has no link
        TypeError
            If graph is not a networkx.Graph
        References
        ----------
        .. 1. Newman, M.E.J. & Girvan, M. Finding and evaluating community
        structure in networks. Physical Review E 69, 26113(2004).
        """
        if nxgraph.is_directed():
            raise TypeError("Bad graph type, use only non directed graph")

        inc = dict([])
        deg = dict([])
        links = nxgraph.size(weight=weight)
        if links == 0:
            raise ValueError("A graph without link has an undefined modularity")

        for node in nxgraph:
            com = partition[node]
            deg[com] = deg.get(com, 0.) + nxgraph.degree(node, weight=weight)
            for neighbor, datas in nxgraph[node].items():
                edge_weight = datas.get(weight, 1)
                if partition[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.) + float(edge_weight)
                    else:
                        inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

        res = 0.
        for com in set(partition.values()):
            res += (inc.get(com, 0.) / links) - \
                   (deg.get(com, 0.) / (2. * links)) ** 2
        return res
    
    def compute_one_level(self, threshold=0.001, weight='weight', limit=100):
        """Compute community assignements for one level. Modifies modularity, partition, node_communities, and communities_sum_incidents
        """
        old_modularity = self.modularity - 0.0001
        [delta, node, target_community] = self.gpu.get_max_delta()

        print(f"[~]node is {node}\n[~]com is {target_community}\n[~]max delta is {delta}")
        while delta > threshold and self.modularity > old_modularity and limit > 0:
            #Update modularity, partition, node_communities, and communities_sum_incidents
            print(f"[~]Run {limit}. Assigning based on what we found")

            limit -= 1
            old_modularity = self.modularity
            old_community = self.partition[node]
            
            # Move node from current community to target and account for summation changes. Remove weight of node to node in community
            for neighbor in self.nxgraph[node]:
                wt = self.nxgraph[node][neighbor][weight] if weight in self.nxgraph[node][neighbor] else 1
                if self.node_communities[neighbor] == target_community:
                    self.communities_sum_incidents[old_community] -= wt
                    self.communities_sum_incidents[self.node_communities[neighbor]] -= wt
                elif self.node_communities[neighbor] == old_community:
                    self.communities_sum_incidents[old_community] += wt
                    self.communities_sum_incidents[self.node_communities[neighbor]] += wt
                else:
                    self.communities_sum_incidents[old_community] -= wt
                    self.communities_sum_incidents[target_community] += wt
                                
            print(f'[~]sum community incidents {self.communities_sum_incidents}')

            want = self.modularity + delta
            hyp = self.partition
            hyp[node] = 999
            hmo = self.compute_modularity(hyp,self.nxgraph)
            hmod = self.compute_modularity(hyp,self.nxgraph) + delta
            print(f"[+]mod with node {node} removed {hmo}")
            print(f"[+]removed and added delta {hmod}")
            
            self.node_communities[node] = target_community #Note: sets c data as well
            self.partition[node] = target_community
            
            self.modularity = self.compute_modularity(self.partition,self.nxgraph)

            got = self.compute_modularity(self.partition,self.nxgraph)
            print(f"[~]want through adding {want}\n[~]caculating Q directly {got}")
            
            [delta, node, target_community] = self.gpu.get_max_delta()
            print(f"[~]node is {node}\n[~]com is {target_community}\n[~]max delta is {delta}...")
            
        print(f'[~]Runs left {limit}')
        
            