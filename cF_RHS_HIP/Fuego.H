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
  static void T2Ei(const double T, double Ei[NUM_SPECIES], const double ceoffs_CKUMS[6*NUM_SPECIES], const int tidx)
  {
    CKUMS(T, Ei, ceoffs_CKUMS, tidx);
  }

  __host__ __device__
  __forceinline__
  static void RTY2Ei(
    const double /*R*/,
    const double T,
    const double* /*Y[]*/,
    double Ei[NUM_SPECIES],
    const double ceoffs_CKUMS[6*NUM_SPECIES],
    const int tidx)
  {
    T2Ei(T, Ei, ceoffs_CKUMS, tidx);
  }
};

#endif