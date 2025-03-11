#ifndef REACTORUTILS_H
#define REACTORUTILS_H

#include "hip/hip_runtime.h"

#include "Fuego.H"

struct YCOrder
{
  // default: [ [Y_0,...Y_n]_{cell 0}, ..., [Y_0,...Y_n]_{cell m}]
  using ordering_type = YCOrder;
};

struct CYOrder
{
  // [ [Y_{cell 0},...Y_m]_{species 0}, ..., [Y_{cell 0},...Y_m]_{species n}]
  using ordering_type = CYOrder;
};

template <typename OrderType>
__host__ __device__ __forceinline__ int
vec_index(const int specie_idx, const int icell, const int /*ncells*/)
{
  return icell * (NUM_SPECIES + 1) + specie_idx;
}

template <typename OrderType>
__device__ __forceinline__ int
spec_index(const int specie_idx, const int icell, const int /*ncells*/)
{
  return icell * NUM_SPECIES + specie_idx;
}

template <>
__host__ __device__ __forceinline__ int
vec_index<CYOrder>(const int specie_idx, const int icell, const int ncells)
{
  return specie_idx * ncells + icell;
}

template <>
__device__ __forceinline__ int
spec_index<CYOrder>(const int specie_idx, const int icell, const int ncells)
{
  return vec_index<CYOrder>(specie_idx, icell, ncells);
}

template <typename OrderType>
__global__ void 
__launch_bounds__(64)
cF_RHS_HIP(
  const int ncells,
  const double dt_save,
  const double* yvec_d,
  double* ydot_d,
  const double* rhoe_init,
  const double* rhoesrc_ext,
  const double* rYs)
{
  const int num_cells_per_block = 1;
  const int lcl_cell_idx = threadIdx.x / 64;
  const int glb_cell_idx = blockIdx.x * num_cells_per_block + lcl_cell_idx;
  const int species_idx = threadIdx.x % 64;

  if (glb_cell_idx < ncells) {

    // has the same effect as if the arrays were 2D
    __shared__ double massfrac[num_cells_per_block * 64];
    __shared__ double ei_pt[num_cells_per_block * 64];
    __shared__ double cdots_pt[num_cells_per_block * 64];
    __shared__ double scratch[num_cells_per_block * 64];

    scratch[threadIdx.x] = 0.0;

    if (species_idx < NUM_SPECIES) {
      massfrac[threadIdx.x] =
        yvec_d[vec_index<OrderType>(species_idx, glb_cell_idx, ncells)]; // requires species index as first argument
      scratch[threadIdx.x] = massfrac[threadIdx.x];
    }

    for (int stride = 32; stride > 0; stride /= 2) {
      if (species_idx < stride) {
          scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      }
    }

    // this "should" be a broadcast
    double rho_pt = scratch[lcl_cell_idx * 64]; // assign first value of every cell
    double rho_pt_inv = 1.0 / rho_pt;
    
    if (species_idx < NUM_SPECIES) {
      massfrac[threadIdx.x] *= rho_pt_inv;
    }

    const double nrg_pt =
      (rhoe_init[glb_cell_idx] + rhoesrc_ext[glb_cell_idx] * dt_save) * rho_pt_inv;

    double temp_pt =
      yvec_d[vec_index<OrderType>(NUM_SPECIES, glb_cell_idx, ncells)];

    double Cv_pt = 0.0;
    struct Fuego eos;
    if (species_idx == 0) {
      // eos.REY2T(rho_pt, nrg_pt, &massfrac[lcl_cell_idx * 64], temp_pt);
      // eos.RTY2Cv(rho_pt, temp_pt, &massfrac[lcl_cell_idx * 64], Cv_pt);
    }
    
    eos.RTY2Ei(rho_pt, temp_pt, &massfrac[lcl_cell_idx * 64], &ei_pt[lcl_cell_idx * 64], species_idx);

    if (species_idx < NUM_SPECIES) {
      scratch[threadIdx.x] =
        cdots_pt[threadIdx.x] +
        rYs[spec_index<OrderType>(species_idx, glb_cell_idx, ncells)];

      ydot_d[vec_index<OrderType>(species_idx, glb_cell_idx, ncells)] =
        scratch[threadIdx.x];

      scratch[threadIdx.x] *= ei_pt[threadIdx.x];
    }

    for (int stride = 32; stride > 0; stride /= 2) {
      if (species_idx < stride) {
          scratch[threadIdx.x] += scratch[threadIdx.x + stride];
      }
    }

    if (species_idx == 0) {
      ydot_d[vec_index<OrderType>(NUM_SPECIES, glb_cell_idx, ncells)] =
        (rhoesrc_ext[glb_cell_idx] - scratch[lcl_cell_idx * 64]) * (rho_pt_inv / Cv_pt);
    }
  }
}

#endif
