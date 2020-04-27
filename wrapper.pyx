import numpy as np
import networkx as nx
cimport numpy as np

# assert sizeof(int) == sizeof(np.int32_t) 

# Using cython to bring in class constructor and attributes
cdef extern from "src/gpucommunity.hpp":
    cdef cppclass C_GPUCommunity "GPUCommunity":
        C_GPUCommunity(int, np.int32_t*, np.int32_t*, np.int32_t*, np.int32_t*, int, int, np.int32_t*, np.int32_t*, np.int32_t*, int)
        void getMaxDelta(double*, int*, int*)

cdef class GPUCommunity:
    cdef C_GPUCommunity* community_state 

    def __cinit__(self,
        np.ndarray[ndim=1, dtype=np.int32_t] node_degrees,
        np.ndarray[ndim=1, dtype=np.int32_t] node_communities,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_incidents,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_inside,
        nxgraph):
        """
        This takes in arrays of data about a graph to hold and update community data 
        
        To be passed to this C++ class constructor
        GPUCommunity::GPUCommunity (
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
        # Note creation of cdef ndarray buffers are allowed only locally, not as member vars
        
        print("[~] GPUCommunity")        
        
        # extract adj matrix in compressed sparse row
        csr = nx.to_scipy_sparse_matrix(nxgraph)
        csr_num_rows = len(csr.indptr) - 1
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_data = np.array(csr.data, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indices = np.array(csr.indices, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indptr = np.array(csr.indptr, dtype=np.int32)
        sum_all_weights = sum(csr.data)/2 # assumes undirected graph (and symmetric adj matrix
        
        # initialize community state in c/cuda       
        
        self.community_state = new C_GPUCommunity(
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
            
    def __dealloc__(self):
        del self.community_state
    
        
    def get_max_delta(self):
        """Parallel computation
        """
        cdef double max
        cdef int node
        cdef int community
        self.community_state.getMaxDelta(&max, &node, &community)
        return [max, node, community]
    
    def reinit_data(self,
        np.ndarray[ndim=1, dtype=np.int32_t] node_degrees,
        np.ndarray[ndim=1, dtype=np.int32_t] node_communities,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_incidents,
        np.ndarray[ndim=1, dtype=np.int32_t] communities_sum_inside,
        nxgraph):

        
        # first dealloc existing data
        del self.community_state
        
        # extract adj matrix in compressed sparse row. CSR already exists in nxgraph
        csr = nx.to_scipy_sparse_matrix(nxgraph)
        csr_num_rows = len(csr.indptr) - 1
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_data = np.array(csr.data, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indices = np.array(csr.indices, dtype=np.int32)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] csr_indptr = np.array(csr.indptr, dtype=np.int32)
        sum_all_weights = sum(csr.data)/2 # assumes undirected graph (and symmetric adj matrix)
        
        # initialize community state in c/cuda
        self.community_state = new C_GPUCommunity(
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

            
class Community():
    def __init__(self, nxgraph, weight='weight'):
        print("[~] Community")
        graph_len = len(nxgraph)
        self.node_degrees = np.array([d for n,d in nxgraph.degree(nxgraph.nodes(), weight=weight)], dtype=np.int32) 
        
        # init all nodes to their own singular communities
        # communities are labeled sequentially
        self.node_communities = np.array(nxgraph.nodes(), dtype=np.int32)

        # initial sum of incident links to communities is same as node's degrees
        # if no attr named 'weight' then defaults to link existence (0 or 1)
        self.communities_sum_incidents = np.array(self.node_degrees, dtype=np.int32) 
      
        # initial sum of internal links is 0
        self.loops = {}
        self.communities_sum_inside = np.zeros(graph_len, dtype=np.int32)
        
        self.nxgraph = nxgraph #current graph
        self.community_graphs = []
        self.community_levels = len(self.community_graphs)
        
        self.gpu = GPUCommunity(
            self.node_degrees,
            self.node_communities,
            self.communities_sum_incidents,
            self.communities_sum_inside,
            self.nxgraph)
        
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
        Adapted from https://github.com/taynaud/python-louvain/
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

    def __renumber(self, dictionary):
        """Renumber the values of the dictionary from 0 to n
        """
        count = 0
        ret = dictionary.copy()
        new_values = dict()

        for key in dictionary.keys():
            value = dictionary[key]
            new_value = new_values.get(value, -1)
            if new_value == -1:
                new_values[value] = count
                new_value = count
                count += 1
            ret[key] = new_value

        return ret    

    def merge_communities(self, weight='weight'):
        """ Adds existing graph to end of list and creates a new member graph and reinitializes member array data. Assumes partition communities have been renumbered from 0, 1, ..., n
        """
        self.community_graphs.extend(self.nxgraph)
        self.community_levels = len(self.community_graphs)
        X = nx.Graph()
        X.add_nodes_from(list(set(self.partition.values())))
        for (no,ne,w) in self.nxgraph.edges(data=True):
            if not w or weight not in w:
                w = 1
            else:# 'weight' in w:
                w = w[weight]
            com_no = self.partition[no]
            com_ne = self.partition[ne]
            if X.has_edge(com_no,com_ne): 
                w += 1 if weight not in X[com_no][com_ne] else X[com_no][com_ne][weight]
            X.add_edge(com_no, com_ne,weight=w)        
        
        self.nxgraph = X.copy()
        self.node_degrees = np.array([d for n,d in self.nxgraph.degree(self.nxgraph.nodes(), weight=weight)], dtype=np.int32)

        self.node_communities = np.array(list(self.nxgraph.nodes()), dtype=np.int32)  # assume renumbered
        self.partition = dict(enumerate(self.node_communities))

        self.communities_sum_incidents = np.array(self.node_degrees, dtype=np.int32)

        #self loops were created from internal sums, so include in new array
        self.communities_sum_inside = np.zeros(len(self.communities_sum_incidents), dtype=np.int32)

        for node in list(nx.nodes_with_selfloops(self.nxgraph)):
            edge_data = self.nxgraph.get_edge_data(node,node,default={weight:0})
            self.loops[node] = edge_data.get(weight, 1)
            self.communities_sum_inside[node] = edge_data[weight]
        
        self.gpu.reinit_data(
            self.node_degrees,
            self.node_communities,
            self.communities_sum_incidents,
            self.communities_sum_inside,
            self.nxgraph)
        
    def compute_one_level(self, threshold=0.0001, weight='weight', limit=10):
        """Compute community assignements for one level. Modifies modularity, partition, node_communities, and communities_sum_incidents
        """
        old_modularity = self.modularity
        
        [delta, node, target_community] = self.gpu.get_max_delta()
        modified = False
        if delta > 0:
            modified = True  #fudge


        #print(f"[~]move node {node} to com {target_community} for max delta {delta}")
        while modified and limit > 0:
            #Update modularity, partition, node_communities, and communities_sum_incidents

            limit -= 1
            old_node = node
            old_community = self.partition[node]
            old_modularity = self.modularity
            
            # Move node from current community to target and account for summation changes
            self.__remove(node, old_community, self.__sum_node_2_com(node, old_community))
            self.__insert(node, target_community, self.__sum_node_2_com(node, target_community))
            
            self.node_communities[node] = target_community #Note: sets c data as well
            self.partition[node] = target_community
            
            self.modularity = self.compute_modularity(self.partition,self.nxgraph)

            [delta, node, target_community] = self.gpu.get_max_delta()

            #print(f"[~]move node {node} to com {target_community} for relmax delta {delta}")

            if self.modularity - old_modularity < threshold:
                break
            
        self.partition = self.__renumber(self.partition)

    def best_partition(self, limit=10, __MIN=0.0005):
        status_list = []
        
        old_mod = self.modularity        
        self.compute_one_level(limit=limit)
        new_mod = self.modularity
        partition = self.partition.copy()
        status_list.append(partition)
        old_mod = new_mod
        self.merge_communities()
        while True:
            self.compute_one_level(limit=limit)
            new_mod = self.modularity
            if new_mod - old_mod < __MIN:
                break
            partition = self.partition.copy()
            status_list.append(partition)
            old_mod = new_mod
            self.merge_communities()
        
        return status_list
        
    def __remove(self, node, community, sum_node_2_com, weight='weight'):
        """ Remove node from community com and modify community degrees and community internal weights"""
        self.communities_sum_incidents[community] -= self.node_degrees[node]
        self.communities_sum_inside[community] = self.communities_sum_inside[community] - sum_node_2_com - self.loops.get(node, 0)
        
        self.node_communities[node] = -1

        
    def __insert(self, node, community, sum_node_2_com, weight='weight'):
        """ Insert node into community and modify community degrees and community internal weights"""
        self.node_communities[node] = community
        self.communities_sum_incidents[community] += self.node_degrees[node]
        self.communities_sum_inside[community] = self.communities_sum_inside[community] + sum_node_2_com + self.loops.get(node, 0)
              
