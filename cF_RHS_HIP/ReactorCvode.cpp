#include <iostream>
#include <stdio.h>

#include "ReactorUtils.H"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

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

int cF_RHS(const int ncells, const double dt_save)
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
  double *yvec_d, *ydot_d, *rhoe_init_d, *rhoesrc_ext_d, *rYsrc_ext_d;

  // initialize arrays
  HIP_ASSERT(hipMalloc((void**)&yvec_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&rhoe_init_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&rhoesrc_ext_d, ncells * sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&rYsrc_ext_d, ncells * NUM_SPECIES * sizeof(double)));

  // copy to device memory
  HIP_ASSERT(hipMemcpy(yvec_d, yvec_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(ydot_d, ydot_h, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoe_init_d, rhoe_init_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rhoesrc_ext_d, rhoesrc_ext_h, ncells * sizeof(double), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(rYsrc_ext_d, rYsrc_ext_h, ncells * NUM_SPECIES * sizeof(double), hipMemcpyHostToDevice));
             
  // kernel call           
  const int block_size = 256;
  cF_RHS_HIP<Ordering>
      <<<(ncells + block_size - 1) / block_size, block_size>>>(
          ncells, dt_save, yvec_d, ydot_d, rhoe_init_d, rhoesrc_ext_d, rYsrc_ext_d);

  hipDeviceSynchronize();

  // copy to host memory
  HIP_ASSERT(hipMemcpy(ydot_h, ydot_d, ncells * (NUM_SPECIES+1) * sizeof(double), hipMemcpyDeviceToHost));

  HIP_ASSERT(hipFree(yvec_d));
  HIP_ASSERT(hipFree(ydot_d));
  HIP_ASSERT(hipFree(rhoe_init_d));
  HIP_ASSERT(hipFree(rhoesrc_ext_d));
  HIP_ASSERT(hipFree(rYsrc_ext_d));

  return 0;
}

int main(int argc, char* argv[]) {
  // Check if exactly 3 arguments (excluding the program name) are passed
  if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3>" << std::endl;
      return 1;
  }

  // Accessing the arguments
  int ncells = std::stoi(argv[1]);
  double time = std::stod(argv[2]);

  return 0;
}