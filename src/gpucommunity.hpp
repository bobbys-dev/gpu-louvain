/*
CPP interface file to define the GPUCommunity class. 
*/

class GPUCommunity {
  // pointer to the GPU memory where the array is stored
  int* array_device;
  // pointer to the CPU memory where the array is stored
  int* array_host;
  // length of the array (number of elements)
  int length;

public:
  /* By using the default names INPLACE_ARRAY1, DIM1 in the header
     file
   */

  GPUCommunity(int* INPLACE_ARRAY1, int DIM1); // constructor (copies to GPU)

  ~GPUCommunity(); // destructor

  void increment(); // does operation inplace on the GPU

  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void retreive_to (int* INPLACE_ARRAY1, int DIM1);


};