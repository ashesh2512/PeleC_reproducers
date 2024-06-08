#include <stdio.h>
#include <iostream>

// error checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Problem parameters
#ifndef NUM_SPECIES
#define NUM_SPECIES 53
#endif
const double MASSFRAC_MIN = -0.00015890965; // based on simulation log
const double MASSFRAC_MAX = 0.73261661;     // based on simulation log
#ifndef NUM_CELLS
#define NUM_CELLS 3145728
#endif

__host__ __device__ int
vec_index(const int specie_idx, const int icell, const int ncells)
{
  return specie_idx * ncells + icell;
}

__global__ void
fKernelSpec_CUDAReg(
  const int ncells,
  const double* yvec_d,
  double* ydot_d)
{
  if (blockIdx.x < ncells) {

    __shared__ double massfrac[NUM_SPECIES];
    __shared__ double scratch[64];

    scratch[threadIdx.x] = 0.0;

    if (threadIdx.x < NUM_SPECIES) {
      massfrac[threadIdx.x] =
        yvec_d[vec_index(threadIdx.x, blockIdx.x, ncells)];
    }
    __syncthreads();

    if (threadIdx.x < NUM_SPECIES) {
      scratch[threadIdx.x] = massfrac[threadIdx.x];
    }
    __syncthreads();

    // // this "should" be a broadcast
    double rho_pt = scratch[0];
    double rho_pt_inv = 1.0 / rho_pt;

    if(threadIdx.x==0 && blockIdx.x==1) {
      printf("scratch[0] before: %16.16f \n", scratch[0]);
    }

    massfrac[threadIdx.x] *= rho_pt_inv;
    __syncthreads(); // before next usage of massfrac

    if(threadIdx.x==0 && blockIdx.x==1) {
      printf("scratch[0] after: %16.16f \n", scratch[0]);
      printf("rho_pt_inv after: %16.16f \n", rho_pt);
    }

    if (threadIdx.x < NUM_SPECIES) {
      ydot_d[vec_index(threadIdx.x, blockIdx.x, ncells)] =
        scratch[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      ydot_d[vec_index(NUM_SPECIES, blockIdx.x, ncells)] = 1.0;
    }
  }
}

__global__ void
fKernelSpec_CUDAOpt(
  const int ncells,
  const double* yvec_d,
  double* ydot_d)
{
  const int num_cells_per_block = 1;
  const int lcl_cell_idx = threadIdx.x / 64;
  const int glb_cell_idx = blockIdx.x * num_cells_per_block + lcl_cell_idx;
  const int lcl_cell_thread_idx = threadIdx.x % 64;

  if (glb_cell_idx < ncells) {

    __shared__ double massfrac[num_cells_per_block][NUM_SPECIES];
    __shared__ double scratch[num_cells_per_block][64];

    scratch[lcl_cell_idx][lcl_cell_thread_idx] = 0.0;

    if (lcl_cell_thread_idx < NUM_SPECIES) {
      massfrac[lcl_cell_idx][lcl_cell_thread_idx] =
        yvec_d[vec_index(lcl_cell_thread_idx, glb_cell_idx, ncells)];
    }
    __syncthreads();

    if (lcl_cell_thread_idx < NUM_SPECIES) {
      scratch[lcl_cell_idx][lcl_cell_thread_idx] = massfrac[lcl_cell_idx][lcl_cell_thread_idx];
    }
    __syncthreads();

    // this "should" be a broadcast
    double rho_pt = scratch[lcl_cell_idx][0];
    double rho_pt_inv = 1.0 / rho_pt;

    if(lcl_cell_thread_idx==0 && glb_cell_idx==1) {
      printf("scratch[lcl_cell_idx][0] before: %16.16f \n", scratch[lcl_cell_idx][0]);
    }

    massfrac[lcl_cell_idx][lcl_cell_thread_idx] *= rho_pt_inv;
    __syncthreads(); // before next usage of massfrac

    if(lcl_cell_thread_idx==0 && glb_cell_idx==1) {
      printf("scratch[lcl_cell_idx][0] after: %16.16f \n", scratch[lcl_cell_idx][0]);
      printf("rho_pt_inv after: %16.16f \n", rho_pt);
    }

    if (lcl_cell_thread_idx < NUM_SPECIES) {
      ydot_d[vec_index(lcl_cell_thread_idx, glb_cell_idx, ncells)] =
        scratch[lcl_cell_idx][lcl_cell_thread_idx];
    }
    __syncthreads();

    if (lcl_cell_thread_idx == 0) {
      ydot_d[vec_index(NUM_SPECIES, glb_cell_idx, ncells)] = 1.0;
    }
  }
}

double random_number(const double lower_bound, const double upper_bound) {
  const long max_rand = 1000000L;
  return lower_bound +
         (upper_bound - lower_bound) * (random() % max_rand) / max_rand;
}

// host wrapper
int main() {

  // Use the values of a, b, and c in your program
  printf(
      "Problem parameters: NUM_SPECIES = %d, NUM_CELLS = %d\n",
      NUM_SPECIES, NUM_CELLS);

  const int num_vals = NUM_CELLS*(NUM_SPECIES+1); //+1 is for temperature

  double *massfrac_h = new double[num_vals];
  double *output_h_reg = new double[num_vals];
  double *output_h_opt = new double[num_vals];

  // create random arrays
  rand();
  for (int i = 0; i < num_vals; i++) {
    massfrac_h[i] = random_number(MASSFRAC_MIN, MASSFRAC_MAX);
    output_h_reg[i] = 0.0;
    output_h_opt[i] = 0.0;
  }

  // allocate memory
  double *massfrac_d, *output_d_reg, *output_d_opt;
  cudaMalloc(&massfrac_d,
             num_vals * sizeof(double)); // allocate device space for massfrac
  cudaMalloc(&output_d_reg,
             num_vals * sizeof(double));
  cudaMalloc(&output_d_opt,
             num_vals * sizeof(double));
  cudaCheckErrors("cudaMalloc failure");

  // copy to device
  cudaMemcpy(massfrac_d, massfrac_h, num_vals * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(output_d_reg, output_h_reg, num_vals * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(output_d_opt, output_h_opt, num_vals * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy HostToDevice failure");

  // launch kernels
  {
    const int nthreads_per_block = 64; // multiple of warpSize rounded up,
                                      // based on number of species
    dim3 block(nthreads_per_block);
    dim3 grid(NUM_CELLS); // 1 cell is assigned 1 block
    fKernelSpec_CUDAReg<<<grid, block>>>(NUM_CELLS, massfrac_d, output_d_reg);

  }
  cudaDeviceSynchronize();
  cudaCheckErrors("fKernelSpec_CUDAReg kernel execution failure");

  {
    const int num_cells_per_block = 1;
    const int nthreads_per_block = 64*num_cells_per_block; // multiple of warpSize rounded up,
                                        // based on number of species
    dim3 block(nthreads_per_block);
    dim3 grid((NUM_CELLS + num_cells_per_block - 1) / num_cells_per_block); // multiple cells assigned 1 block
    fKernelSpec_CUDAOpt<<<grid, block>>>(NUM_CELLS, massfrac_d, output_d_opt);

  }
  cudaDeviceSynchronize();
  cudaCheckErrors("fKernelSpec_CUDAOpt kernel execution failure");

  // copy to host
  cudaMemcpy(output_h_reg, output_d_reg, NUM_CELLS * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy DeviceToHost failure for output_d_reg");

  cudaMemcpy(output_h_opt, output_d_opt, NUM_CELLS * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy DeviceToHost failure for output_d_opt");

  // check results
  double base = 0.0;
  double opt = 0.0;
  double diff = 0.0;
  double tol = 1e-1;
  for (int i = 0; i < NUM_CELLS; i++) {
    for (int n = 0; n < (NUM_SPECIES + 1); n++) {
      base = output_h_reg[vec_index(n, i, NUM_CELLS)];
      opt = output_h_opt[vec_index(n, i, NUM_CELLS)];
      diff = base - opt;

      if ((std::abs(diff) > tol) && (std::abs(diff / base) > tol)) {
        printf(
          "Base: %16.16f, Opt: %16.16f, Abs Diff: %16.16f, Rel Diff: %16.16f, "
          "for cell %d, for species %d \n",
          base, opt, std::abs(diff), std::abs(diff / base), i, n);
        std::cout<<("Someone messed up the computations.");
        exit(0);
      }
    }
  }

  return 0;
}
