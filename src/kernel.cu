#include <stdio.h>

static __device__ int d_state = 0;

static __device__ void lock(void) { while(atomicCAS(&d_state, 0, 1) != 0); }  // this only works on the latest GPUs (Volta and newer)

static __device__ void unlock(void) { d_state = 0; }

static __device__ double foo(void) {
    return threadIdx.x * 1.0;
}

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
    for (int i = start; csr_indptr[i] != csr_nnz && i < stop; i++) {
        int neighbor = csr_indices[i];
        if(node_communities[neighbor] == target_community) {
            sum += csr_data[i];
        }
    }
    return sum;
}

__global__ void kernel_add_one(int* a, int length) {
    /**
    Kernel used to debug arrays
    **/
    int gid = threadIdx.x + blockDim.x * blockIdx.x;

    while(gid < length) {
    	a[gid] += 1;
        gid += blockDim.x * gridDim.x;
    }
}

__global__ void kernel_init_fixed_values(const int nodes_count, int node_degrees[], const int csr_nnz, const int csr_data[], const int csr_indptr[]) {
    /**
    This should be called only once to calculate values that will persist for lifetime of graph. Assumes undirected graph.
    node_degrees[]
    **/
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (gid < nodes_count) {
        node_degrees[gid] = 0;
        // loop thru csr indices given at csr_indptr[gid] to csr_indptr[gid+1]
        // be sure to also exit if csr_indptr[gid] reaches the end given by nnz
        for (int i = csr_indptr[gid]; csr_indptr[i] != csr_nnz && i < csr_indptr[gid+1]; i++) {
            node_degrees[gid] += csr_data[i];
        }
    }
}

__global__ void kernel_compute_max_delta(const int nodes_count, const int node_communities[], const int node_degrees[], const int communities_sum_incidents[], const int communities_sum_inside[], const int csr_nnz, const int csr_num_rows, const int csr_data[], const int csr_indices[], const int csr_indptr[], const int sum_all_weights, int* const best_global_node, int* const best_global_community, double* max_global_delta) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x; //gid is effectively node id    
    double saw = sum_all_weights *2; 
    
    // create and initialize a shared local val per block to share local max
    __shared__ unsigned long long val;
    if (threadIdx.x == 0) val = 0;
    __syncthreads();
    
    if (gid < nodes_count) {
        int best_local_node = gid; // gid = this node
        int best_local_community = node_communities[gid]; //best community ground truth

        double max_local_delta = 0;
        // TODO 1.1: add int my_community = node_communities[gid]; here
        int ex_community = node_communities[gid];
        int my_temp_community = -1; // this should cause the if to always run
        //Iterate through each neighbor and calc delta if in a diff community
        for (int i = csr_indptr[gid]; csr_indptr[i] != csr_nnz && i < csr_indptr[gid+1]; i++) {
            int neighbor = csr_indices[i];            
            int target_community = node_communities[neighbor];
            //TODO 1.2 change node_communities[gid] to my_community. Check that it runs same results as baseline
            if (my_temp_community != target_community) {
                int sum_node_to_community = get_sum_node_to_community(node_communities, csr_nnz, csr_num_rows, csr_data, csr_indices, csr_indptr, gid, target_community);
                
                //TODO 2: remove the 2* from term 1 both in conjunction and seperate from TODO 1
                double term1 = 2 * sum_node_to_community / saw;
                //TODO 1.3: modify communities_sum_incidents addition to check if when original community, then have to add additional weights of sum of incidents as if the node were outside the original community
                double term2a = 0.0;
                if (ex_community == target_community) {
                    term2a = (double)sum_node_to_community;
                }
                double term2 = (2 * (communities_sum_incidents[target_community] + term2a) * (node_degrees[gid])) / (saw * saw);
                double some_delta = term1 - term2;
                if (some_delta > max_local_delta) {
                    max_local_delta = some_delta;
                    best_local_community = target_community;
                }
            }
        } // end neighborcommunity loop
        
        // Find max among thread in block, then check for global max, if so, update best node and community
        // Range of max_local_delta = [0, 1.0]
        // Use shifted value to find max among threads in block. 
        unsigned long long loc = max_local_delta * 0x08000000; // mult to preserve some precision. Recommended factor less than 2^27
        loc = (loc << 32) + threadIdx.x; // shift out of threadIdx.x space
        atomicMax(&val, loc); // block max
        __syncthreads();

         //only one thread per block will run the following code
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

}
