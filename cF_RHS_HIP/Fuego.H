#ifndef FUEGO_H
#define FUEGO_H

#include "hip/hip_runtime.h"

#include "mechanism.H"
// #include "PhysicsConstants.H"

struct Fuego
{
  using eos_type = Fuego;

  static std::string identifier() { return "Fuego"; }

  __host__ __device__
  __forceinline__
  static void
  TY2Cv(const double T, const double Y[NUM_SPECIES], double& Cv)
  {
    CKCVBS(T, Y, Cv);
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
    GET_T_GIVEN_EY(E, Y, T, lierr);
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
    CKYTCR(R, T, Y, C);
    CKWC(T, C, WDOT);

    for (int n = 0; n < NUM_SPECIES; n++) {
      WDOT[n] *= mw(n);
    }
  }

  __host__ __device__
  __forceinline__
  static void T2Ei(const double T, double Ei[NUM_SPECIES])
  {
    CKUMS(T, Ei);
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