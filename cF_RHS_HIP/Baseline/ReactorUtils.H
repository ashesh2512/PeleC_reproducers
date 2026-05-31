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
__launch_bounds__(256)
cF_RHS_HIP(
  const int ncells,
  const double dt_save,
  const double* yvec_d,
  double* ydot_d,
  const double* rhoe_init,
  const double* rhoesrc_ext,
  const double* rYs)
{
#if USE_GRID_STRIDE
  for (long tidx = long(blockDim.x)*blockIdx.x+threadIdx.x, stride = long(blockDim.x)*gridDim.x; tidx < long(ncells); tidx += stride)
#else
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= ncells)
    return;
#endif
  {
    double rho_pt = 0.0;
    double massfrac[NUM_SPECIES] = {0.0};
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = yvec_d[vec_index<OrderType>(n, tidx, ncells)];
      rho_pt += massfrac[n];
    }
    const double rho_pt_inv = 1.0 / rho_pt;

    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] *= rho_pt_inv;
    }

    const double nrg_pt =
      (rhoe_init[tidx] + rhoesrc_ext[tidx] * dt_save) * rho_pt_inv;

    double temp_pt = yvec_d[vec_index<OrderType>(NUM_SPECIES, tidx, ncells)];

    double Cv_pt = 0.0;
    double ei_pt[NUM_SPECIES] = {0.0};

    struct Fuego eos;
    eos.REY2T(rho_pt, nrg_pt, massfrac, temp_pt);
    eos.RTY2Ei(rho_pt, temp_pt, massfrac, ei_pt);
    eos.RTY2Cv(rho_pt, temp_pt, massfrac, Cv_pt);

    double cdots_pt[NUM_SPECIES] = {0.0};
    eos.RTY2WDOT(rho_pt, temp_pt, massfrac, cdots_pt);

    double rhoesrc = rhoesrc_ext[tidx];
    for (int n = 0; n < NUM_SPECIES; n++) {
      const double cdot_rYs =
        cdots_pt[n] + rYs[spec_index<OrderType>(n, tidx, ncells)];
      ydot_d[vec_index<OrderType>(n, tidx, ncells)] = cdot_rYs;
      rhoesrc -= cdot_rYs * ei_pt[n];
    }
    ydot_d[vec_index<OrderType>(NUM_SPECIES, tidx, ncells)] =
      rhoesrc * (rho_pt_inv / Cv_pt);
  }
}

#endif
