/*
This is the central piece of code. This file implements a class
(interface in gpucommunity.hpp) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (ie compute value of modularity delta) that let
you perform actions with the GPU
This class will get translated into python via cython

CUDA C/C++ keyword __global__ indicates a function that runs on device, but is called from host code. Host launches a kernel

*/

#include <gpucommunity.hpp>
#include <assert.h>
#include <iostream>
#include "kernel.cu"

using namespace std;

static const int THREADS_PER_BLOCK = 512;

static void CheckCuda() {
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

GPUCommunity::GPUCommunity (const int nodes_count, int node_communities[], int node_degrees[], int communities_sum_incidents[], int communities_sum_inside[], const int csr_nnz, const int csr_num_rows, int csr_data[], int csr_indices[], int csr_indptr[], int sum_all_weights) {
  /*
  Initialize all member variables including:
  nodes, sum of weights incident to nodes, community ids, sum of weights inside communities, sum of weights internal to community, sum of all graph weights, 
  and compressed sparse row (CSR) matrix for adjacency matrix 
  */
  this->verbose = false;
  this->nodes_count = nodes_count; // number of nodes
  this->sum_all_weights = sum_all_weights; // sum all weights in graph
  this->csr_nnz = csr_nnz; //non-zero values in compressed matrix
  this->csr_num_rows = csr_num_rows; // number of rows in original adjacency matrix
  this->node_communities = node_communities;
  this->node_degrees = node_degrees;
  this->communities_sum_incidents = communities_sum_incidents;
  this->communities_sum_inside = communities_sum_inside;
  this->csr_data = csr_data;
  this->csr_indices = csr_indices;
  this->csr_indptr = csr_indptr;
  
  // nodes community assignments
  if (cudaSuccess != cudaMalloc((void**)&d_node_communities, sizeof(int)*nodes_count)) {fprintf(stderr, "ERROR: could not allocate community memory\n"); exit(-1);}
  
  // sum of links incident to node 
  if (cudaSuccess != cudaMalloc((void**)&d_node_degrees, sizeof(int)*nodes_count)) {fprintf(stderr, "ERROR: could not allocate node memory\n"); exit(-1);}

  // sum of weights incident to community
  if (cudaSuccess != cudaMalloc((void**)&d_communities_sum_incidents, sizeof(int)*nodes_count)) {fprintf(stderr, "ERROR: could not allocate community memory\n"); exit(-1);}
  
  // sum of weights inside communities
  if (cudaSuccess != cudaMalloc((void**)&d_communities_sum_inside, sizeof(int)*nodes_count)) {fprintf(stderr, "ERROR: could not allocate community memory\n"); exit(-1);}

  // compressed sparse row matrix and graph state
   if (cudaSuccess != cudaMalloc((void**)&d_csr_data, sizeof(int)*csr_nnz)) {fprintf(stderr, "ERROR: could not allocate CSR data in memory\n"); exit(-1);}
  
  if (cudaSuccess != cudaMalloc((void**)&d_csr_indices, sizeof(int)*csr_nnz)) {fprintf(stderr, "ERROR: could not allocate CSR data in memory\n"); exit(-1);}

  // size of number of rows + 1 because nnz is at the end
  if (cudaSuccess != cudaMalloc((void**)&d_csr_indptr, sizeof(int)*(csr_num_rows+1))) {fprintf(stderr, "ERROR: could not allocate CSR data in memory\n"); exit(-1);}

  // shared sum of all link weights. TODO: remove this?
  if (cudaSuccess != cudaMalloc((void**)&d_sum_all_weights, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate graph state memory\n"); exit(-1);}
  
}

void GPUCommunity::printState() {
  
    printf("[+]Nodes count: %d\n",nodes_count);
    printf("csr_nnz: %d:\n",csr_nnz);
    printf("csr_num_rows: %d\n",csr_num_rows);
    printf("sum weights in graph: %d\n",sum_all_weights);

    if (verbose) {
        printf("\n[+]Node:community assignments:\n");
        for (int i = 0; i < nodes_count; i++) {
            printf("%d:%d, ", i, node_communities[i]);
        }
        
        printf("\n[+]Node degrees:\n");
        for (int i = 0; i < nodes_count; i++) {
            printf("%d ",this->node_degrees[i]);
        }
        
        printf("\n[+]Sum of weights incident to communities:\n");
        for (int i = 0; i < nodes_count; i++) {
            printf("%d ",communities_sum_incidents[i]);
        }
    
        printf("\n[+]Sum of weights internal to community:\n");   
        for (int i = 0; i < nodes_count; i++) {
            printf("%d ",communities_sum_inside[i]);
        }
        
        printf("\n[+]CSR data:\n");
        for (int i = 0; i < csr_nnz; i++) {
            printf("%d ",csr_data[i]);
        }
        printf("\n[+]CSR indices:\n");
        for (int i = 0; i < csr_nnz; i++) {
            printf("%d ",csr_indices[i]);
        }
        
        // 0 to nnz+1 because nnz value is appended in indptr array
        printf("\n[+]CSR index pointer:\n");
        for (int i = 0; i <= csr_num_rows ; i++) {
            printf("%d ",csr_indptr[i]);
        }
    }
    
    printf("\n\n");

}
void GPUCommunity::getMaxDelta (double* max, int* node, int* community) {
    /*
    This method internal state of nodes and communities and computes delta modularity. Kernel needs node community assignments, community weight info, adjacency info. Each loop there will bee a best node and best assignment
    Invariant: node (and its state) for which deltas are being computed
    Input: array of communities
    Output: community reassignment that will lead to max delta
    */
    
    int best_node;
    int* d_best_node;
    int best_community;
    int* d_best_community;
    double max_delta = 0.;
    double* d_max_delta;

    
    // allocate a global device variables that will hold best node and best corresponding community assignment. These variables are paired and shouldn't be treated separate
    if (cudaSuccess != cudaMalloc((void **)&d_best_node, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

    if (cudaSuccess != cudaMalloc((void **)&d_best_community, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
    
    if (cudaSuccess != cudaMalloc((void **)&d_max_delta, sizeof(double))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_max_delta, &max_delta, sizeof(double), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_node_degrees, node_degrees, sizeof(int)*nodes_count, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_csr_data, csr_data, sizeof(int)*csr_nnz, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_csr_indices, csr_indices, sizeof(int)*csr_nnz, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_csr_indptr, csr_indptr, sizeof(int)*(csr_num_rows+1), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
    
    //Note: following arrays are updated each loop to device
    
    if (cudaSuccess != cudaMemcpy(d_node_communities, node_communities, sizeof(int)*nodes_count, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
    
    if (cudaSuccess != cudaMemcpy(d_communities_sum_incidents, communities_sum_incidents, sizeof(int)*nodes_count, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(d_communities_sum_inside, communities_sum_inside, sizeof(int)*nodes_count, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
    
    kernel_compute_max_delta_b<<<(nodes_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(nodes_count, d_node_communities, d_node_degrees, d_communities_sum_incidents,  d_communities_sum_inside, csr_nnz, csr_num_rows, d_csr_data, d_csr_indices, d_csr_indptr, sum_all_weights, d_best_node, d_best_community, d_max_delta);
    
    CheckCuda();
    
    if (cudaSuccess != cudaMemcpy(&best_node, d_best_node, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying best_node to host failed\n"); exit(-1);}

    if (cudaSuccess != cudaMemcpy(&best_community, d_best_community, sizeof(int), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying best_community to host failed\n"); exit(-1);}
    
    if (cudaSuccess != cudaMemcpy(&max_delta, d_max_delta, sizeof(double), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying max_delta to host failed\n"); exit(-1);}
    
    *node = best_node;
    *community = best_community;
    *max = max_delta; 

    cudaFree(d_best_node);
    cudaFree(d_best_community);
    cudaFree(d_max_delta);
}


GPUCommunity::~GPUCommunity() {
    cudaFree(d_node_communities);
    cudaFree(d_node_degrees);
    cudaFree(d_communities_sum_incidents);
    cudaFree(d_communities_sum_inside);
    cudaFree(d_csr_data);
    cudaFree(d_csr_indices);
    cudaFree(d_csr_indptr);
    cudaFree(d_sum_all_weights);
    printf("[+] CUDA freed!! ;-)\n");
}


