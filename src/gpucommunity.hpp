/*
CPP interface file to define the GPUCommunity class. Objects of this class will have dual host and device arrays. The device arrays will be allocated on an instantiation.
*/

#ifndef GPUCOMMUNITY_H
#define GPUCOMMUNITY_H

class GPUCommunity {

  /*
  Relevant arrays
  */

  //nodes community assignments
  int nodes_count;
  
  // shared summation
  int sum_all_weights;
  int* d_sum_all_weights;
  
  int* node_communities;
  int* d_node_communities;

  // sum of links incident to node. length = nodes_count
  int* node_degrees;
  int* d_node_degrees;
  
  // sum of weights incident to community. length = nodes_count
  int* communities_sum_incidents;
  int* d_communities_sum_incidents;
  
  // sum of weights inside communities. length = nodes_count
  int* communities_sum_inside;
  int* d_communities_sum_inside;

  // Allocate compressed sparse row matrix and graph state
  // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
  int csr_nnz;
  int csr_num_rows;
  
  //length = csr_nnz
  int* csr_data;
  int* d_csr_data;
  
  //length = csr_nnz
  int* csr_indices;
  int* d_csr_indices;

  // length = csr_num_rows + 1 because nnz is append at end
  int* csr_indptr;
  int* d_csr_indptr;

  bool verbose; 
  
public:

  GPUCommunity (const int nodes_count, int node_communities[], int node_degrees[], int communities_sum_incidents[], int communities_sum_inside[], const int nnz, const int csr_num_rows, int csr_data[], int csr_indices[], int csr_indptr[], int sum_all_weights); // constructor (copies to GPU)

  ~GPUCommunity(); // destructor
  void printState();
  void getMaxDelta (double* max, int* node, int* community);

};

#endif