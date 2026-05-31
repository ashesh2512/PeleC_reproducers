#include <assert.h>
#include <iostream>
#include <stdio.h>

#include "hip/hip_runtime.h"

#include "ReactorUtils.H"
#include "ReactorUtils_opt.H"

#include <cmath>

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

// PeleC compile time paramters
using Ordering = CYOrder;

// TODO: could change the input to be exactly what is in PeleC
// by printing it out during a PeleC run, and reading from that file
// for this reproducer
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
  double* ydot_opt_h = new double[ncells * (NUM_SPECIES+1)];
  for (int i = 0; i < (ncells * NUM_SPECIES); i++) {
    yvec_h[i] = random_number(MASSFRAC_MIN, MASSFRAC_MAX);
    ydot_h[i] = 0.0;
    ydot_opt_h[i] = 0.0;
  }
  for (int i = (ncells * NUM_SPECIES); i < (ncells * (NUM_SPECIES+1)); i++) {
    yvec_h[i] = random_number(T_MIN, T_MAX);
    ydot_h[i] = 0.0;
    ydot_opt_h[i] = 0.0;
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
  double *ydot_d_opt = nullptr;

  // initialize arrays
  HIP_ASSERT(hipMalloc(&yvec_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc(&ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc(&ydot_d_opt, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rhoe_init_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rhoesrc_ext_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc(&rYsrc_ext_d, ncells * NUM_SPECIES * sizeof(double)));

  // copy to device memory
  HIP_ASSERT(hipMemcpy(yvec_d, yvec_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(ydot_d, ydot_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(ydot_d_opt, ydot_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoe_init_d, rhoe_init_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoesrc_ext_d, rhoesrc_ext_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rYsrc_ext_d, rYsrc_ext_h, ncells * NUM_SPECIES * sizeof(double), hipMemcpyHostToDevice));

  printf("\nKernels about to be launched, data copied over to device.\n");
  fflush(stdout);

  for (int iter = 0; iter < num_iters; iter ++)
  {
    // kernel call (baseline)
    const int block_size = 256;
    cF_RHS_HIP<Ordering>
        <<<(ncells + block_size - 1) / block_size, block_size>>>(
            ncells, dt_save, yvec_d, ydot_d, rhoe_init_d, rhoesrc_ext_d, rYsrc_ext_d);

    // kernel call (optimized)
    cF_RHS_HIP_opt<Ordering>
        <<<(ncells + block_size - 1) / block_size, block_size>>>(
            ncells, dt_save, yvec_d, ydot_d_opt, rhoe_init_d, rhoesrc_ext_d, rYsrc_ext_d);
  }

  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
      fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
  }

  // copy to host memory
  HIP_ASSERT(hipMemcpy(ydot_h, ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(ydot_opt_h, ydot_d_opt, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyDeviceToHost));

  printf("\nKernels launched, completed, and data copied from device.\n");
  fflush(stdout);

  // --- Regression test: compare ydot_d vs ydot_d_opt elementwise ---
  int mismatches = 0;
  double max_abs_err = 1e-15;
  double max_rel_err = 1e-15;
  int max_abs_idx = -1;
  int max_rel_idx = -1;
  const int total_elems = ncells * (NUM_SPECIES + 1);

  for (int i = 0; i < total_elems; i++) {
    const double val = ydot_h[i];
    const double val_opt = ydot_opt_h[i];
    const double abs_err = fabs(val - val_opt);
    const double denom = fmax(fabs(val), 1.0e-300);
    const double rel_err = abs_err / denom;

    if (abs_err > 0.0) {
      mismatches++;
    }
    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      max_abs_idx = i;
    }
    if (rel_err > max_rel_err) {
      max_rel_err = rel_err;
      max_rel_idx = i;
    }
  }

  printf("\n=== Regression Test: ydot_d vs ydot_d_opt ===\n");
  printf("Total elements: %d\n", total_elems);
  printf("Mismatches (nonzero abs error): %d\n", mismatches);
  printf("Max absolute error: %.15e (at index %d)\n", max_abs_err, max_abs_idx);
  printf("Max relative error: %.15e (at index %d)\n", max_rel_err, max_rel_idx);
  if (mismatches == 0) {
    printf("PASSED: ydot_d and ydot_d_opt are bitwise identical.\n");
  } else {
    printf("INFO: ydot_d and ydot_d_opt differ at %d elements.\n", mismatches);
  }
  printf("================================================\n\n");

  HIP_ASSERT(hipFree(yvec_d));
  HIP_ASSERT(hipFree(ydot_d));
  HIP_ASSERT(hipFree(ydot_d_opt));
  HIP_ASSERT(hipFree(rhoe_init_d));
  HIP_ASSERT(hipFree(rhoesrc_ext_d));
  HIP_ASSERT(hipFree(rYsrc_ext_d));

  delete[] yvec_h;
  delete[] ydot_h;
  delete[] ydot_opt_h;
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