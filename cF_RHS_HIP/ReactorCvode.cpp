#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "hip/hip_runtime.h"

#include "ReactorUtils.H"

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

// PeleC compile time paramters
using Ordering = CYOrder;

// Problem parameters
const double DENSITY_MIN = 0.00018493468; // based on simulation log
const double DENSITY_MAX = 0.0012255019; // based on simulation log

const double MASSFRAC_MIN = -0.00015890965; // based on simulation log
const double MASSFRAC_MAX = 0.73261661;     // based on simulation log

const double T_MIN = 297.99515;             // based on simulation log
const double T_MAX = 1891.6578;             // based on simulation log

double random_number(const double lower_bound, const double upper_bound) {
  const long max_rand = 1000000L;
  return lower_bound +
         (upper_bound - lower_bound) * (random() % max_rand) / max_rand;
}

int cF_RHS(const int ncells, const double dt_save, const int num_iters)
{
  // create and populate arrays
  double* yvec_h = new double[ncells * (NUM_SPECIES+1)];
  double* ydot_h = new double[ncells * (NUM_SPECIES+1)];
  for (int i = 0; i < (ncells * NUM_SPECIES); i++) {
    yvec_h[i] = random_number(MASSFRAC_MIN, MASSFRAC_MAX);
    ydot_h[i] = 0.0;
  }
  for (int i = (ncells * NUM_SPECIES); i < (ncells * (NUM_SPECIES+1)); i++) {
    yvec_h[i] = random_number(T_MIN, T_MAX);
    ydot_h[i] = 0.0;
  }

  double* rhoe_init_h = new double[ncells];
  double* rhoesrc_ext_h = new double[ncells];
  for (int i = 0; i < ncells; i++) {
    rhoe_init_h[i] = random_number(DENSITY_MIN, DENSITY_MAX);
    rhoesrc_ext_h[i] = 0.0;
  }

  double* rYsrc_ext_h = new double[ncells * NUM_SPECIES];
  for (int i = 0; i < (ncells * NUM_SPECIES); i++) {
    rYsrc_ext_h[i] = random_number(MASSFRAC_MIN, MASSFRAC_MAX);
  }

  // initialize device pointers and copy over to them
  double *yvec_d = nullptr, *ydot_d = nullptr, *rhoe_init_d = nullptr, *rhoesrc_ext_d = nullptr, *rYsrc_ext_d = nullptr;

  // initialize arrays
  HIP_ASSERT(hipMalloc(&yvec_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc(&ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rhoe_init_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rhoesrc_ext_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rYsrc_ext_d, ncells * NUM_SPECIES * sizeof(double)));

  // copy to device memory
  HIP_ASSERT(hipMemcpy(yvec_d, yvec_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(ydot_d, ydot_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoe_init_d, rhoe_init_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoesrc_ext_d, rhoesrc_ext_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rYsrc_ext_d, rYsrc_ext_h, ncells * NUM_SPECIES * sizeof(double), hipMemcpyHostToDevice));

  printf("\nKernel about to be launched, data copied over to device.\n");
  fflush(stdout);

  {
    double* mat_7_h = new double[7*NUM_SPECIES]; // allocate space for dummy mat
    double* mat_6_h = new double[6*NUM_SPECIES]; // allocate space for dummy mat

    // create random arrays
    for (int i = 0; i < 7*NUM_SPECIES; i++) {
      mat_7_h[i] = random_number(1e-15, 10.0);
    }
    for (int i = 0; i < 6*NUM_SPECIES; i++) {
      mat_6_h[i] = random_number(1e-15, 10.0);
    }

    double* mat_7_d; 
    double* mat_6_d;
    HIP_ASSERT(hipMalloc(&mat_7_d, 7*NUM_SPECIES * sizeof(double))); // allocate device space for dummy mat
    HIP_ASSERT(hipMalloc(&mat_6_d, 6*NUM_SPECIES * sizeof(double))); // allocate device space for dummy mat

    HIP_ASSERT(hipMemcpy(mat_7_d, mat_7_h, 7*NUM_SPECIES * sizeof(double), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(mat_6_d, mat_6_h, 6*NUM_SPECIES * sizeof(double), hipMemcpyHostToDevice));

    const int num_cells_per_block = 4;
    const int nthreads_per_block = 64 * num_cells_per_block;

    dim3 block(nthreads_per_block);
    dim3 grid( (ncells + num_cells_per_block - 1) / num_cells_per_block); // multiple cells assigned 1 block

    for (int iter = 0; iter < num_iters; iter ++)
    {
      cF_RHS_HIP<Ordering><<<grid, block>>>(
        ncells, dt_save, yvec_d, ydot_d, rhoe_init_d, rhoesrc_ext_d,
        rYsrc_ext_d, mat_7_d, mat_6_d, mat_6_d);
    }

    HIP_ASSERT(hipFree(mat_7_d));
    HIP_ASSERT(hipFree(mat_6_d));
    
    delete[] mat_7_h; 
    delete[] mat_6_h;
  }

  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
      fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
  }

  // copy to host memory
  HIP_ASSERT(hipMemcpy(ydot_h, ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyDeviceToHost));

  printf("\nKernel launched, completed, and data copied over fromt device. End of program.\n");
  fflush(stdout);

  HIP_ASSERT(hipFree(yvec_d));
  HIP_ASSERT(hipFree(ydot_d));
  HIP_ASSERT(hipFree(rhoe_init_d));
  HIP_ASSERT(hipFree(rhoesrc_ext_d));
  HIP_ASSERT(hipFree(rYsrc_ext_d));
  
  delete[] yvec_h;
  delete[] ydot_h;
  delete[] rhoe_init_h;
  delete[] rhoesrc_ext_h;
  delete[] rYsrc_ext_h;

  return 0;
}

int main(int argc, char* argv[]) {

  // Set default values
  std::string arg1 = "131072";
  std::string arg2 = "1.0";
  std::string arg3 = "10000";

  // Check if arguments were provided
  if (argc > 1) {
    arg1 = argv[1];
  }
  printf("Num cells = %s\n", arg1.c_str());

  if (argc > 2) {
    arg2 = argv[2];
  }
  printf("Time step = %s\n", arg2.c_str());

  if (argc > 3) {
    arg3 = argv[3];
  }
  printf("Number of iterations = %s\n", arg3.c_str());

  // Accessing the arguments
  int ncells = std::stoi(arg1);
  double time = std::stod(arg2);
  int num_iters = std::stoi(arg3);

  cF_RHS(ncells, time, num_iters);

  return 0;
}