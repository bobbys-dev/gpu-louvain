#include <stdio.h>

static __device__ int d_state = 0;

static __device__ void lock(void) { while(atomicCAS(&d_state, 0, 1) != 0); }  // this only works on the latest GPUs (Volta and newer)

static __device__ void unlock(void) { d_state = 0; }

static __device__ int get_sum_node_to_community(const int node_communities[], const int csr_nnz, const int csr_num_rows, const int csr_data[], const int csr_indices[], const int csr_indptr[], const int gid, const int target_community) {
    /**
    Device function called by a node that gives sum of links to provided target community.
    Invariant: gid of thread is <= csr num rows
    **/
    
    int sum = 0;
    // loop thru csr indices given at csr_indptr[gid] to csr_indptr[gid+1]
    // if neighbor is in target community, add to sum
    // be sure to also exit if csr_indptr[gid] reaches the end given by nnz
    const int start = csr_indptr[gid];
    const int stop =  csr_indptr[gid+1];
    for (int i = start; i < csr_nnz && i < stop; i++) {
        int neighbor = csr_indices[i];
        if(neighbor != gid && node_communities[neighbor] == target_community) {
            sum += csr_data[i];
        }
    }
    return sum;
}

__global__ void kernel_compute_max_delta_b(const int nodes_count, const int node_communities[], const int node_degrees[], const int communities_sum_incidents[], const int communities_sum_inside[], const int csr_nnz, const int csr_num_rows, const int csr_data[], const int csr_indices[], const int csr_indptr[], const int sum_all_weights, volatile int* const best_global_node, volatile int* const best_global_community, volatile double* max_global_delta) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x; //gid is effectively node id    
    __shared__ unsigned long long int val;
    int best_local_node = gid; // gid == this node
    int best_local_community = 0;
    double max_local_delta = 0;
    if (threadIdx.x == 0) val = 1025; //Init to val that threadIdX can't be
    __syncthreads();

    // Find local max delta
    if (gid < nodes_count) {
        int com_node = node_communities[gid];
        best_local_community = com_node;//node_communities[gid]; //best community ground truth
        
        double degc_totw = node_degrees[gid]/((double)sum_all_weights * 2.0);
        
        double sum_node_to_own_c = get_sum_node_to_community(node_communities, csr_nnz, csr_num_rows, csr_data, csr_indices, csr_indptr, gid, com_node);
        double remove_cost = -sum_node_to_own_c + (communities_sum_incidents[com_node] - node_degrees[gid]) * degc_totw;
        
        int ex_com_node = com_node;
        com_node = -1;
        //From here on have to use modified sum_inside and sum_incident for target community
        
        //Iterate through my neighbors
        int start = csr_indptr[gid];
        int stop = csr_indptr[gid+1];
        for (int i = start; i < csr_nnz && i < stop; i++) {
            int neighbor = csr_indices[i];
            int target_community = node_communities[neighbor];
            
            double com_deg_mod = 0.0;
            
            if (target_community == ex_com_node) {
                com_deg_mod = (double)node_degrees[gid]; 
            }
            
            double dnc = get_sum_node_to_community(node_communities, csr_nnz, csr_num_rows, csr_data, csr_indices, csr_indptr, gid, target_community);
            
            double incr = remove_cost + dnc - ((communities_sum_incidents[target_community] - com_deg_mod)*degc_totw);
            
            if (incr > max_local_delta) {
                max_local_delta = incr;
                best_local_community = target_community;
            }
        }
        
        // Find max among thread in block, then check for global max, if so, update best node and community
        // Use shifted value to find max among threads in block. 
        // create and initialize a shared local val per block to share local max
        unsigned long long int loc = max_local_delta * 0x08000000; // mult to preserve some precision. Recommended factor less than 2^27
        loc = (loc << 32) + threadIdx.x; // shift out of threadIdx.x space        
        atomicMax(&val, loc); // block max
    } // End if

    __syncthreads();
           
    // Only one thread per block will run the following code
    if (threadIdx.x == (val & 0xffffffff)) {
        if (max_local_delta > *max_global_delta) {
            lock();
            if (max_local_delta > *max_global_delta) {
                *max_global_delta = max_local_delta;
                *best_global_node = best_local_node;
                *best_global_community = best_local_community;
            }
            unlock();
        }
    } 

}

