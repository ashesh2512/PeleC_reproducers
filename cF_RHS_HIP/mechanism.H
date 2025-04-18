#ifndef MECHANISM_H
#define MECHANISM_H

#include "hip/hip_runtime.h"

/* Elements
0  O
1  H
2  C
3  N
*/

// Species
#define NC12H26_ID 0
#define H_ID 1
#define O_ID 2
#define OH_ID 3
#define HO2_ID 4
#define H2_ID 5
#define H2O_ID 6
#define H2O2_ID 7
#define O2_ID 8
#define CH2_ID 9
#define CH2D_ID 10
#define CH3_ID 11
#define CH4_ID 12
#define HCO_ID 13
#define CH2O_ID 14
#define CH3O_ID 15
#define CO_ID 16
#define CO2_ID 17
#define C2H2_ID 18
#define C2H3_ID 19
#define C2H4_ID 20
#define C2H5_ID 21
#define C2H6_ID 22
#define CH2CHO_ID 23
#define aC3H5_ID 24
#define C3H6_ID 25
#define nC3H7_ID 26
#define C2H3CHO_ID 27
#define C4H7_ID 28
#define C4H81_ID 29
#define pC4H9_ID 30
#define C5H9_ID 31
#define C5H10_ID 32
#define PXC5H11_ID 33
#define C6H12_ID 34
#define PXC6H13_ID 35
#define C7H14_ID 36
#define PXC7H15_ID 37
#define C8H16_ID 38
#define PXC8H17_ID 39
#define C9H18_ID 40
#define PXC9H19_ID 41
#define C10H20_ID 42
#define PXC10H21_ID 43
#define PXC12H25_ID 44
#define SXC12H25_ID 45
#define S3XC12H25_ID 46
#define C12H24_ID 47
#define C12H25O2_ID 48
#define C12OOH_ID 49
#define O2C12H24OOH_ID 50
#define OC12H23OOH_ID 51
#define N2_ID 52

#define NUM_GAS_ELEMENTS 4    // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 53    // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 268 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  inverse molecular weights
#ifdef USE_GPU
__constant__ const double global_imw[53] = {
  0.0058706117177410, // NC12H26
  0.9920634920634921, // H
  0.0625039064941559, // O
  0.0587993179279120, // OH
  0.0302975216627280, // HO2
  0.4960317460317460, // H2
  0.0555092978073827, // H2O
  0.0293996589639560, // H2O2
  0.0312519532470779, // O2
  0.0712910814857061, // CH2
  0.0712910814857061, // CH2*
  0.0665114732291320, // CH3
  0.0623324814560868, // CH4
  0.0344613688055690, // HCO
  0.0333044694598015, // CH2O
  0.0322227234645872, // CH3O
  0.0357015351660121, // CO
  0.0227226249176305, // CO2
  0.0384054074813734, // C2H2
  0.0369740442209569, // C2H3
  0.0356455407428531, // C2H4
  0.0344091941366733, // C2H5
  0.0332557366145660, // C2H6
  0.0232315019165989, // CH2CHO
  0.0243468945535997, // aC3H5
  0.0237636938285687, // C3H6
  0.0232077792476038, // nC3H7
  0.0178367579908676, // C2H3CHO
  0.0181488203266788, // C4H7
  0.0178227703714265, // C4H81
  0.0175082288675678, // pC4H9
  0.0144661275623128, // C5H9
  0.0142582162971412, // C5H10
  0.0140561966743039, // PXC5H11
  0.0118818469142844, // C6H12
  0.0117412234354820, // PXC6H13
  0.0101844402122437, // C7H14
  0.0100809500287307, // PXC7H15
  0.0089113851857133, // C8H16
  0.0088320497421041, // PXC8H17
  0.0079212312761896, // C9H18
  0.0078584844126962, // PXC9H19
  0.0071291081485706, // C10H20
  0.0070782428969833, // PXC10H21
  0.0059055583114828, // PXC12H25
  0.0059055583114828, // SXC12H25
  0.0059055583114828, // S3XC12H25
  0.0059409234571422, // C12H24
  0.0049669696518154, // C12H25O2
  0.0049669696518154, // C12OOH
  0.0042858122471371, // O2C12H24OOH
  0.0046227596950828, // OC12H23OOH
  0.0356964374955379, // N2
};
#endif
const double h_global_imw[53] = {
  0.0058706117177410, // NC12H26
  0.9920634920634921, // H
  0.0625039064941559, // O
  0.0587993179279120, // OH
  0.0302975216627280, // HO2
  0.4960317460317460, // H2
  0.0555092978073827, // H2O
  0.0293996589639560, // H2O2
  0.0312519532470779, // O2
  0.0712910814857061, // CH2
  0.0712910814857061, // CH2*
  0.0665114732291320, // CH3
  0.0623324814560868, // CH4
  0.0344613688055690, // HCO
  0.0333044694598015, // CH2O
  0.0322227234645872, // CH3O
  0.0357015351660121, // CO
  0.0227226249176305, // CO2
  0.0384054074813734, // C2H2
  0.0369740442209569, // C2H3
  0.0356455407428531, // C2H4
  0.0344091941366733, // C2H5
  0.0332557366145660, // C2H6
  0.0232315019165989, // CH2CHO
  0.0243468945535997, // aC3H5
  0.0237636938285687, // C3H6
  0.0232077792476038, // nC3H7
  0.0178367579908676, // C2H3CHO
  0.0181488203266788, // C4H7
  0.0178227703714265, // C4H81
  0.0175082288675678, // pC4H9
  0.0144661275623128, // C5H9
  0.0142582162971412, // C5H10
  0.0140561966743039, // PXC5H11
  0.0118818469142844, // C6H12
  0.0117412234354820, // PXC6H13
  0.0101844402122437, // C7H14
  0.0100809500287307, // PXC7H15
  0.0089113851857133, // C8H16
  0.0088320497421041, // PXC8H17
  0.0079212312761896, // C9H18
  0.0078584844126962, // PXC9H19
  0.0071291081485706, // C10H20
  0.0070782428969833, // PXC10H21
  0.0059055583114828, // PXC12H25
  0.0059055583114828, // SXC12H25
  0.0059055583114828, // S3XC12H25
  0.0059409234571422, // C12H24
  0.0049669696518154, // C12H25O2
  0.0049669696518154, // C12OOH
  0.0042858122471371, // O2C12H24OOH
  0.0046227596950828, // OC12H23OOH
  0.0356964374955379, // N2
};

//  molecular weights
#ifdef USE_GPU
__constant__ const double global_mw[53] = {
  170.340000, // NC12H26
  1.008000,   // H
  15.999000,  // O
  17.007000,  // OH
  33.006000,  // HO2
  2.016000,   // H2
  18.015000,  // H2O
  34.014000,  // H2O2
  31.998000,  // O2
  14.027000,  // CH2
  14.027000,  // CH2*
  15.035000,  // CH3
  16.043000,  // CH4
  29.018000,  // HCO
  30.026000,  // CH2O
  31.034000,  // CH3O
  28.010000,  // CO
  44.009000,  // CO2
  26.038000,  // C2H2
  27.046000,  // C2H3
  28.054000,  // C2H4
  29.062000,  // C2H5
  30.070000,  // C2H6
  43.045000,  // CH2CHO
  41.073000,  // aC3H5
  42.081000,  // C3H6
  43.089000,  // nC3H7
  56.064000,  // C2H3CHO
  55.100000,  // C4H7
  56.108000,  // C4H81
  57.116000,  // pC4H9
  69.127000,  // C5H9
  70.135000,  // C5H10
  71.143000,  // PXC5H11
  84.162000,  // C6H12
  85.170000,  // PXC6H13
  98.189000,  // C7H14
  99.197000,  // PXC7H15
  112.216000, // C8H16
  113.224000, // PXC8H17
  126.243000, // C9H18
  127.251000, // PXC9H19
  140.270000, // C10H20
  141.278000, // PXC10H21
  169.332000, // PXC12H25
  169.332000, // SXC12H25
  169.332000, // S3XC12H25
  168.324000, // C12H24
  201.330000, // C12H25O2
  201.330000, // C12OOH
  233.328000, // O2C12H24OOH
  216.321000, // OC12H23OOH
  28.014000,  // N2
};
#endif
const double h_global_mw[53] = {
  170.340000, // NC12H26
  1.008000,   // H
  15.999000,  // O
  17.007000,  // OH
  33.006000,  // HO2
  2.016000,   // H2
  18.015000,  // H2O
  34.014000,  // H2O2
  31.998000,  // O2
  14.027000,  // CH2
  14.027000,  // CH2*
  15.035000,  // CH3
  16.043000,  // CH4
  29.018000,  // HCO
  30.026000,  // CH2O
  31.034000,  // CH3O
  28.010000,  // CO
  44.009000,  // CO2
  26.038000,  // C2H2
  27.046000,  // C2H3
  28.054000,  // C2H4
  29.062000,  // C2H5
  30.070000,  // C2H6
  43.045000,  // CH2CHO
  41.073000,  // aC3H5
  42.081000,  // C3H6
  43.089000,  // nC3H7
  56.064000,  // C2H3CHO
  55.100000,  // C4H7
  56.108000,  // C4H81
  57.116000,  // pC4H9
  69.127000,  // C5H9
  70.135000,  // C5H10
  71.143000,  // PXC5H11
  84.162000,  // C6H12
  85.170000,  // PXC6H13
  98.189000,  // C7H14
  99.197000,  // PXC7H15
  112.216000, // C8H16
  113.224000, // PXC8H17
  126.243000, // C9H18
  127.251000, // PXC9H19
  140.270000, // C10H20
  141.278000, // PXC10H21
  169.332000, // PXC12H25
  169.332000, // SXC12H25
  169.332000, // S3XC12H25
  168.324000, // C12H24
  201.330000, // C12H25O2
  201.330000, // C12OOH
  233.328000, // O2C12H24OOH
  216.321000, // OC12H23OOH
  28.014000,  // N2
};

//  inverse molecular weight
__host__ __device__ __forceinline__ double
imw(const int n)
{
#if USE_GPU
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}

//  molecular weight
__host__ __device__ __forceinline__ double
mw(const int n)
{
#if USE_GPU
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

// compute the e/(RT) at the given temperature
__host__ __device__ __forceinline__ void
speciesInternalEnergy(double* species, const double T, const double ceoffs_CKUMS[6 * NUM_SPECIES], const int tidx)
{
  const double T2 = T * T;
  const double T3 = T * T * T;
  const double T4 = T * T * T * T;
  const double invT = 1.0 / T;

  if (tidx < NUM_SPECIES) {

    const double a = ceoffs_CKUMS[tidx];
    const double b = ceoffs_CKUMS[NUM_SPECIES + tidx];
    const double c = ceoffs_CKUMS[2 * NUM_SPECIES + tidx];
    const double d = ceoffs_CKUMS[3 * NUM_SPECIES + tidx];
    const double e = ceoffs_CKUMS[4 * NUM_SPECIES + tidx];
    const double f = ceoffs_CKUMS[5 * NUM_SPECIES + tidx];

    species[tidx] = a - b * T + c * T2 - d * T3 + e * T4 + f * invT;

    // species[tidx] = ceoffs_CKUMS[tidx] - ceoffs_CKUMS[NUM_SPECIES + tidx] * T +
    //              ceoffs_CKUMS[2 * NUM_SPECIES + tidx] * T2 -
    //              ceoffs_CKUMS[3 * NUM_SPECIES + tidx] * T3 +
    //              ceoffs_CKUMS[4 * NUM_SPECIES + tidx] * T4 +
    //              ceoffs_CKUMS[5 * NUM_SPECIES + tidx] * invT;

    // species[tidx] = ceoffs_CKUMS[6*tidx] - ceoffs_CKUMS[6*tidx + 1] * T +
    //              ceoffs_CKUMS[6*tidx + 2] * T2 -
    //              ceoffs_CKUMS[6*tidx + 3] * T3 +
    //              ceoffs_CKUMS[6*tidx + 4] * T4 +
    //              ceoffs_CKUMS[6*tidx + 5] * invT;
  }
}

// Returns the mean specific heat at CV (Eq. 36)
__host__ __device__ __forceinline__ void
CKCVBS(
  const double T, 
  const double y[], 
  double& cvbs,
  double scratch[],
  const double ceoffs_CKCVBS[6 * NUM_SPECIES],
  const int tidx)
{
  double result = 0.0;
  // compute Cv/R at the given temperature
  const double T2 = T * T;
  const double T3 = T * T * T;
  const double T4 = T * T * T * T;

  if (tidx < NUM_SPECIES) {

    const double a = ceoffs_CKCVBS[tidx];
    const double b = ceoffs_CKCVBS[NUM_SPECIES + tidx];
    const double c = ceoffs_CKCVBS[2 * NUM_SPECIES + tidx];
    const double d = ceoffs_CKCVBS[3 * NUM_SPECIES + tidx];
    const double e = ceoffs_CKCVBS[4 * NUM_SPECIES + tidx];
    const double f = ceoffs_CKCVBS[5 * NUM_SPECIES + tidx];

    scratch[tidx] = y[tidx] * (a + b * T - c * T2 + d * T3 - e * T4 ) * f;

    // scratch[tidx] = y[tidx] *
    //              (ceoffs_CKCVBS[tidx] + ceoffs_CKCVBS[NUM_SPECIES + tidx] * T -
    //               ceoffs_CKCVBS[2 * NUM_SPECIES + tidx] * T2 +
    //               ceoffs_CKCVBS[3 * NUM_SPECIES + tidx] * T3 -
    //               ceoffs_CKCVBS[4 * NUM_SPECIES + tidx] * T4) *
    //              ceoffs_CKCVBS[5 * NUM_SPECIES + tidx];

    // scratch[tidx] =
    //   y[tidx] *
    //   (ceoffs_CKCVBS[6 * tidx] + ceoffs_CKCVBS[6 * tidx + 1] * T -
    //    ceoffs_CKCVBS[6 * tidx + 2] * T2 + ceoffs_CKCVBS[6 * tidx + 3] * T3 -
    //    ceoffs_CKCVBS[6 * tidx + 4] * T4) *
    //   ceoffs_CKCVBS[6 * tidx + 5];
  }

  for (int stride = 32; stride > 0; stride /= 2) {
      if (tidx < stride) {
          scratch[tidx] += scratch[tidx + stride];
      }
  }

  cvbs = scratch[0] * 8.31446261815324e+07;
}

// get mean internal energy in mass units
__host__ __device__ __forceinline__ void
CKUBMS(
  const double T, 
  const double y[],
  double& ubms,
  double scratch[],
  const double ceoffs_CKUBMS[7 * NUM_SPECIES],
  const int tidx)
{
  double result = 0.0;
  const double T2 = T * T;
  const double T3 = T * T * T;
  const double T4 = T * T * T * T;
  const double invT = 1.0 / T;

  if (tidx < NUM_SPECIES) {

    const double a = ceoffs_CKUBMS[tidx];
    const double b = ceoffs_CKUBMS[NUM_SPECIES + tidx];
    const double c = ceoffs_CKUBMS[2 * NUM_SPECIES + tidx];
    const double d = ceoffs_CKUBMS[3 * NUM_SPECIES + tidx];
    const double e = ceoffs_CKUBMS[4 * NUM_SPECIES + tidx];
    const double f = ceoffs_CKUBMS[5 * NUM_SPECIES + tidx];
    const double g = ceoffs_CKUBMS[6 * NUM_SPECIES + tidx];

    scratch[tidx] = y[tidx] * (a + b * T - c * T2 + d * T3 - e * T4 + f * invT) * g;

    // scratch[tidx] = y[tidx] *
    //              (ceoffs_CKUBMS[tidx] + ceoffs_CKUBMS[NUM_SPECIES + tidx] * T -
    //               ceoffs_CKUBMS[2 * NUM_SPECIES + tidx] * T2 +
    //               ceoffs_CKUBMS[3 * NUM_SPECIES + tidx] * T3 -
    //               ceoffs_CKUBMS[4 * NUM_SPECIES + tidx] * T4 +
    //               ceoffs_CKUBMS[5 * NUM_SPECIES + tidx] * invT) *
    //              ceoffs_CKUBMS[6 * NUM_SPECIES + tidx];

    // scratch[tidx] = y[tidx] *
    //              (ceoffs_CKUBMS[7*tidx] + ceoffs_CKUBMS[7*tidx + 1] * T -
    //               ceoffs_CKUBMS[7*tidx + 2] * T2 +
    //               ceoffs_CKUBMS[7*tidx + 3] * T3 -
    //               ceoffs_CKUBMS[7*tidx + 4] * T4 +
    //               ceoffs_CKUBMS[7*tidx + 5] * invT) *
    //              ceoffs_CKUBMS[7*tidx + 6];
  }

  for (int stride = 32; stride > 0; stride /= 2) {
      if (tidx < stride) {
          scratch[tidx] += scratch[tidx + stride];
      }
  }

  const double RT = 8.31446261815324e+07 * T; // R*T

  ubms = scratch[0] * RT;
}

//  get temperature given internal energy in mass units and mass fracs
__host__ __device__ __forceinline__ void
GET_T_GIVEN_EY(
    const double e,
    const double y[],
    double scratch[64],
    const double ceoffs_CKUBMS[7 * NUM_SPECIES],
    const double ceoffs_CKCVBS[6 * NUM_SPECIES],
    double &t,
    int &ierr,
    const int tidx)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const double tol = 1.e-12;
#else
  // const int maxiter = 200;
  const int maxiter = 3; // because we are feeding in garbage
  const double tol = 1.e-6;
#endif
  double tmin = 90;   // max lower bound for thermo def
  double tmax = 4000; // min upper bound for thermo def
  double e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin, scratch, ceoffs_CKUBMS, tidx);
  CKUBMS(tmax, y, emax, scratch, ceoffs_CKUBMS, tidx);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv, scratch, ceoffs_CKCVBS, tidx);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmin, y, cv, scratch, ceoffs_CKCVBS, tidx);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(tmin, y, emin, scratch, ceoffs_CKUBMS, tidx);
    CKCVBS(tmin, y, cv, scratch, ceoffs_CKCVBS, tidx);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Returns internal energy in mass units (Eq 30.)
__host__ __device__ __forceinline__ void
CKUMS(const double T, double ums[], const double ceoffs_CKUMS[6 * NUM_SPECIES], const int tidx)
{
  speciesInternalEnergy(ums, T, ceoffs_CKUMS, tidx);

  if (tidx < NUM_SPECIES) {
    ums[tidx] *= 8.31446261815324e+07 * T * imw(tidx);
  }
}

#endif
