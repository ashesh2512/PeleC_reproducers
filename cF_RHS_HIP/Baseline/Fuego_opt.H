#ifndef FUEGO_OPT_H
#define FUEGO_OPT_H

#include "hip/hip_runtime.h"

#include "mechanism_opt.H"

struct Fuego_opt
{
  using eos_type = Fuego_opt;

  static std::string identifier() { return "Fuego_opt"; }

  __host__ __device__
  __forceinline__
  static void
  TY2Cv(const double T, const double Y[NUM_SPECIES], double& Cv)
  {
    CKCVBS_opt(T, Y, Cv);
  }

  __host__ __device__
  __forceinline__
  static void RTY2Cv(
    const double /*R*/,
    const double T,
    const double Y[NUM_SPECIES],
    double& Cv)
  {
    TY2Cv(T, Y, Cv);
  }

  __host__ __device__
  __forceinline__
  static void
  EY2T(const double E, const double Y[NUM_SPECIES], double& T)
  {
    int lierr = 0;
    GET_T_GIVEN_EY_opt(E, Y, T, lierr);
  }

  __host__ __device__
  __forceinline__
  static void REY2T(
    const double /*R*/,
    const double E,
    const double Y[NUM_SPECIES],
    double& T)
  {
    EY2T(E, Y, T);
  }

  __host__ __device__
  __forceinline__
  static void RTY2WDOT(
    const double R,
    const double T,
    const double Y[NUM_SPECIES],
    double WDOT[NUM_SPECIES])
  {
    double C[NUM_SPECIES];
    CKYTCR_opt(R, T, Y, C);
    CKWC_opt(T, C, WDOT);

    for (int n = 0; n < NUM_SPECIES; n++) {
      WDOT[n] *= mw_opt(n);
    }
  }

  __host__ __device__
  __forceinline__
  static void T2Ei(const double T, double Ei[NUM_SPECIES])
  {
    CKUMS_opt(T, Ei);
  }

  __host__ __device__
  __forceinline__
  static void RTY2Ei(
    const double /*R*/,
    const double T,
    const double* /*Y[]*/,
    double Ei[NUM_SPECIES])
  {
    T2Ei(T, Ei);
  }
};

#endif
