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

    const double a = ceoffs_CKUMS[threadIdx.x];
    const double b = ceoffs_CKUMS[NUM_SPECIES + threadIdx.x];
    const double c = ceoffs_CKUMS[2 * NUM_SPECIES + threadIdx.x];
    const double d = ceoffs_CKUMS[3 * NUM_SPECIES + threadIdx.x];
    const double e = ceoffs_CKUMS[4 * NUM_SPECIES + threadIdx.x];
    const double f = ceoffs_CKUMS[5 * NUM_SPECIES + threadIdx.x];

    species[threadIdx.x] = a - b * T + c * T2 - d * T3 + e * T4 + f * invT;

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

// compute the g/(RT) at the given temperature
__host__ __device__ __forceinline__ void
gibbs(double* species, const double T)
{
  const double T2 = T * T;
  const double T3 = T * T * T;
  const double T4 = T * T * T * T;
  const double invT = 1.0 / T;
  const double logT = log(T);

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 1: H
    species[1] = +2.547365990000000e+04 * invT + 2.946682853000000e+00 -
                 2.500000000000000e+00 * logT - 3.526664095000000e-13 * T +
                 3.326532733333333e-16 * T2 - 1.917346933333333e-19 * T3 +
                 4.638661660000000e-23 * T4;
    // species 2: O
    species[2] = +2.912225920000000e+04 * invT + 1.116333640000000e+00 -
                 3.168267100000000e+00 * logT + 1.639659420000000e-03 * T -
                 1.107177326666667e-06 * T2 + 5.106721866666666e-10 * T3 -
                 1.056329855000000e-13 * T4;
    // species 3: OH
    species[3] = +3.381538120000000e+03 * invT + 4.815738570000000e+00 -
                 4.125305610000000e+00 * logT + 1.612724695000000e-03 * T -
                 1.087941151666667e-06 * T2 + 4.832113691666666e-10 * T3 -
                 1.031186895000000e-13 * T4;
    // species 4: HO2
    species[4] = +2.948080400000000e+02 * invT + 5.851355599999999e-01 -
                 4.301798010000000e+00 * logT + 2.374560255000000e-03 * T -
                 3.526381516666666e-06 * T2 + 2.023032450000000e-09 * T3 -
                 4.646125620000001e-13 * T4;
    // species 5: H2
    species[5] = -9.179351730000000e+02 * invT + 1.661320882000000e+00 -
                 2.344331120000000e+00 * logT - 3.990260375000000e-03 * T +
                 3.246358500000000e-06 * T2 - 1.679767450000000e-09 * T3 +
                 3.688058805000000e-13 * T4;
    // species 6: H2O
    species[6] = -3.029372670000000e+04 * invT + 5.047672768000000e+00 -
                 4.198640560000000e+00 * logT + 1.018217050000000e-03 * T -
                 1.086733685000000e-06 * T2 + 4.573308850000000e-10 * T3 -
                 8.859890850000000e-14 * T4;
    // species 7: H2O2
    species[7] = -1.770258210000000e+04 * invT + 8.410619499999998e-01 -
                 4.276112690000000e+00 * logT + 2.714112085000000e-04 * T -
                 2.788928350000000e-06 * T2 + 1.798090108333333e-09 * T3 -
                 4.312271815000000e-13 * T4;
    // species 8: O2
    species[8] = -1.063943560000000e+03 * invT + 1.247806300000001e-01 -
                 3.782456360000000e+00 * logT + 1.498367080000000e-03 * T -
                 1.641217001666667e-06 * T2 + 8.067745908333334e-10 * T3 -
                 1.621864185000000e-13 * T4;
    // species 9: CH2
    species[9] = +4.600404010000000e+04 * invT + 2.200146820000000e+00 -
                 3.762678670000000e+00 * logT - 4.844360715000000e-04 * T -
                 4.658164016666667e-07 * T2 + 3.209092941666667e-10 * T3 -
                 8.437085950000000e-14 * T4;
    // species 10: CH2*
    species[10] = +5.049681630000000e+04 * invT + 4.967723077000000e+00 -
                  4.198604110000000e+00 * logT + 1.183307095000000e-03 * T -
                  1.372160366666667e-06 * T2 + 5.573466508333334e-10 * T3 -
                  9.715736850000000e-14 * T4;
    // species 11: CH3
    species[11] = +1.644499880000000e+04 * invT + 2.069026070000000e+00 -
                  3.673590400000000e+00 * logT - 1.005475875000000e-03 * T -
                  9.550364266666668e-07 * T2 + 5.725978541666666e-10 * T3 -
                  1.271928670000000e-13 * T4;
    // species 12: CH4
    species[12] = -1.024664760000000e+04 * invT + 9.791179889999999e+00 -
                  5.149876130000000e+00 * logT + 6.835489400000000e-03 * T -
                  8.196676650000000e-06 * T2 + 4.039525216666667e-09 * T3 -
                  8.334697800000000e-13 * T4;
    // species 13: HCO
    species[13] = +3.839564960000000e+03 * invT + 8.268134100000002e-01 -
                  4.221185840000000e+00 * logT + 1.621962660000000e-03 * T -
                  2.296657433333333e-06 * T2 + 1.109534108333333e-09 * T3 -
                  2.168844325000000e-13 * T4;
    // species 14: CH2O
    species[14] = -1.430895670000000e+04 * invT + 4.190910250000000e+00 -
                  4.793723150000000e+00 * logT + 4.954166845000000e-03 * T -
                  6.220333466666666e-06 * T2 + 3.160710508333333e-09 * T3 -
                  6.588632600000000e-13 * T4;
    // species 15: CH3O
    species[15] = +1.295697600000000e+03 * invT - 2.860603620000000e+00 -
                  3.711805020000000e+00 * logT + 1.402316530000000e-03 * T -
                  6.275849516666667e-06 * T2 + 3.942267408333333e-09 * T3 -
                  9.329421000000001e-13 * T4;
    // species 16: CO
    species[16] = -1.434408600000000e+04 * invT + 7.112418999999992e-02 -
                  3.579533470000000e+00 * logT + 3.051768400000000e-04 * T -
                  1.694690550000000e-07 * T2 - 7.558382366666667e-11 * T3 +
                  4.522122495000000e-14 * T4;
    // species 17: CO2
    species[17] = -4.837196970000000e+04 * invT - 7.544278700000000e+00 -
                  2.356773520000000e+00 * logT - 4.492298385000000e-03 * T +
                  1.187260448333333e-06 * T2 - 2.049325183333333e-10 * T3 +
                  7.184977399999999e-15 * T4;
    // species 18: C2H2
    species[18] = +2.642898070000000e+04 * invT - 1.313102400600000e+01 -
                  8.086810940000000e-01 * logT - 1.168078145000000e-02 * T +
                  5.919530250000000e-06 * T2 - 2.334603641666667e-09 * T3 +
                  4.250364870000000e-13 * T4;
    // species 19: C2H3
    species[19] = +3.485984680000000e+04 * invT - 5.298073800000000e+00 -
                  3.212466450000000e+00 * logT - 7.573958100000000e-04 * T -
                  4.320156866666666e-06 * T2 + 2.980482058333333e-09 * T3 -
                  7.357543650000000e-13 * T4;
    // species 20: C2H4
    species[20] = +5.089775930000000e+03 * invT - 1.381294799999999e-01 -
                  3.959201480000000e+00 * logT + 3.785261235000000e-03 * T -
                  9.516504866666667e-06 * T2 + 5.763239608333333e-09 * T3 -
                  1.349421865000000e-12 * T4;
    // species 21: C2H5
    species[21] = +1.284162650000000e+04 * invT - 4.007435600000004e-01 -
                  4.306465680000000e+00 * logT + 2.093294460000000e-03 * T -
                  8.285713450000000e-06 * T2 + 4.992721716666666e-09 * T3 -
                  1.152545020000000e-12 * T4;
    // species 22: C2H6
    species[22] = -1.152220550000000e+04 * invT + 1.624601760000000e+00 -
                  4.291424920000000e+00 * logT + 2.750771350000000e-03 * T -
                  9.990638133333334e-06 * T2 + 5.903885708333334e-09 * T3 -
                  1.343428855000000e-12 * T4;
    // species 23: CH2CHO
    species[23] = +6.200000000000000e+01 * invT - 6.162391100000001e+00 -
                  3.409062400000000e+00 * logT - 5.369287000000000e-03 * T -
                  3.152487500000000e-07 * T2 + 5.965485916666667e-10 * T3 -
                  1.433692550000000e-13 * T4;
    // species 24: aC3H5
    species[24] = +1.924562900000000e+04 * invT - 1.581003050000000e+01 -
                  1.363183500000000e+00 * logT - 9.906910499999999e-03 * T -
                  2.082843333333333e-06 * T2 + 2.779629583333333e-09 * T3 -
                  7.923285500000000e-13 * T4;
    // species 25: C3H6
    species[25] = +1.074826000000000e+03 * invT - 1.465203300000000e+01 -
                  1.493307000000000e+00 * logT - 1.046259000000000e-02 * T -
                  7.477990000000000e-07 * T2 + 1.390760000000000e-09 * T3 -
                  3.579073000000000e-13 * T4;
    // species 26: nC3H7
    species[26] = +1.031234600000000e+04 * invT - 2.008691670000000e+01 -
                  1.049117300000000e+00 * logT - 1.300448650000000e-02 * T -
                  3.923752666666667e-07 * T2 + 1.632927666666667e-09 * T3 -
                  4.686010350000000e-13 * T4;
    // species 27: C2H3CHO
    species[27] = -9.335734399999999e+03 * invT - 1.822672720000000e+01 -
                  1.271349800000000e+00 * logT - 1.311552700000000e-02 * T +
                  1.548538416666667e-06 * T2 + 3.986439333333333e-10 * T3 -
                  1.674027150000000e-13 * T4;
    // species 28: C4H7
    species[28] = +2.265332800000000e+04 * invT - 2.269338368000000e+01 -
                  7.444943200000000e-01 * logT - 1.983942850000000e-02 * T +
                  3.816347666666667e-06 * T2 - 1.779414416666667e-10 * T3 -
                  1.154818750000000e-13 * T4;
    // species 29: C4H81
    species[29] = -1.790400400000000e+03 * invT - 1.988133100000000e+01 -
                  1.181138000000000e+00 * logT - 1.542669000000000e-02 * T -
                  8.477541166666667e-07 * T2 + 2.054574000000000e-09 * T3 -
                  5.555096499999999e-13 * T4;
    // species 30: pC4H9
    species[30] = +7.322104000000000e+03 * invT - 2.096056380000000e+01 -
                  1.208704200000000e+00 * logT - 1.914874850000000e-02 * T +
                  1.211008483333333e-06 * T2 + 1.285712250000000e-09 * T3 -
                  4.342971750000000e-13 * T4;
    // species 31: C5H9
    species[31] = +2.812188700000000e+03 * invT - 3.887825510000000e+01 +
                  2.419011100000000e+00 * logT - 2.021519450000000e-02 * T -
                  1.130038983333333e-06 * T2 + 2.810395166666667e-09 * T3 -
                  7.558356500000001e-13 * T4;
    // species 48: C12H25O2
    species[48] = -2.989180000000000e+04 * invT - 1.166006000000000e+01 -
                  5.314040000000000e+00 * logT - 4.469365000000000e-02 * T -
                  2.422516666666667e-06 * T2 + 6.243750000000000e-09 * T3 -
                  1.676625000000000e-12 * T4;
    // species 49: C12OOH
    species[49] = -2.383800000000000e+04 * invT - 1.420029000000000e+01 -
                  5.152310000000000e+00 * logT - 4.989565000000000e-02 * T +
                  3.010583333333333e-06 * T2 + 3.486958333333333e-09 * T3 -
                  1.113930000000000e-12 * T4;
    // species 50: O2C12H24OOH
    species[50] = -4.168750000000000e+04 * invT - 4.086092800000000e+01 -
                  4.819720000000000e-01 * logT - 7.251000000000001e-02 * T +
                  1.665513333333333e-05 * T2 - 2.170183333333333e-09 * T3 -
                  5.967900000000001e-14 * T4;
    // species 51: OC12H23OOH
    species[51] = -6.653610000000001e+04 * invT + 1.965780000000001e+00 -
                  8.807330000000000e+00 * logT - 3.253115000000000e-02 * T -
                  1.158430000000000e-05 * T2 + 1.057541666666667e-08 * T3 -
                  2.554955000000000e-12 * T4;
    // species 52: N2
    species[52] = -1.020899900000000e+03 * invT - 6.516950000000001e-01 -
                  3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                  6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                  1.222427000000000e-13 * T4;
  } else {
    // species 1: H
    species[1] = +2.547365990000000e+04 * invT + 2.946682924000000e+00 -
                 2.500000010000000e+00 * logT + 1.154214865000000e-11 * T -
                 2.692699133333334e-15 * T2 + 3.945960291666667e-19 * T3 -
                 2.490986785000000e-23 * T4;
    // species 2: O
    species[2] = +2.921757910000000e+04 * invT - 2.214917859999999e+00 -
                 2.569420780000000e+00 * logT + 4.298705685000000e-05 * T -
                 6.991409816666667e-09 * T2 + 8.348149916666666e-13 * T3 -
                 6.141684549999999e-17 * T4;
    // species 3: OH
    species[3] = +3.718857740000000e+03 * invT - 2.836911870000000e+00 -
                 2.864728860000000e+00 * logT - 5.282522400000000e-04 * T +
                 4.318045966666667e-08 * T2 - 2.543488950000000e-12 * T3 +
                 6.659793800000000e-17 * T4;
    // species 4: HO2
    species[4] = +1.118567130000000e+02 * invT + 2.321087500000001e-01 -
                 4.017210900000000e+00 * logT - 1.119910065000000e-03 * T +
                 1.056096916666667e-07 * T2 - 9.520530833333334e-12 * T3 +
                 5.395426750000000e-16 * T4;
    // species 5: H2
    species[5] = -9.501589220000000e+02 * invT + 6.542302510000000e+00 -
                 3.337279200000000e+00 * logT + 2.470123655000000e-05 * T -
                 8.324279633333333e-08 * T2 + 1.496386616666667e-11 * T3 -
                 1.001276880000000e-15 * T4;
    // species 6: H2O
    species[6] = -3.000429710000000e+04 * invT - 1.932777610000000e+00 -
                 3.033992490000000e+00 * logT - 1.088459020000000e-03 * T +
                 2.734541966666666e-08 * T2 + 8.086832250000000e-12 * T3 -
                 8.410049600000000e-16 * T4;
    // species 7: H2O2
    species[7] = -1.786178770000000e+04 * invT + 1.248846229999999e+00 -
                 4.165002850000000e+00 * logT - 2.454158470000000e-03 * T +
                 3.168987083333333e-07 * T2 - 3.093216550000000e-11 * T3 +
                 1.439541525000000e-15 * T4;
    // species 8: O2
    species[8] = -1.088457720000000e+03 * invT - 2.170693450000000e+00 -
                 3.282537840000000e+00 * logT - 7.415437700000000e-04 * T +
                 1.263277781666667e-07 * T2 - 1.745587958333333e-11 * T3 +
                 1.083588970000000e-15 * T4;
    // species 9: CH2
    species[9] = +4.626360400000000e+04 * invT - 3.297092110000000e+00 -
                 2.874101130000000e+00 * logT - 1.828196460000000e-03 * T +
                 2.348243283333333e-07 * T2 - 2.168162908333333e-11 * T3 +
                 9.386378350000000e-16 * T4;
    // species 10: CH2*
    species[10] = +5.092599970000000e+04 * invT - 6.334463270000000e+00 -
                  2.292038420000000e+00 * logT - 2.327943185000000e-03 * T +
                  3.353199116666667e-07 * T2 - 3.482550000000000e-11 * T3 +
                  1.698581825000000e-15 * T4;
    // species 11: CH3
    species[11] = +1.677558430000000e+04 * invT - 6.194354070000000e+00 -
                  2.285717720000000e+00 * logT - 3.619950185000000e-03 * T +
                  4.978572466666667e-07 * T2 - 4.964038700000000e-11 * T3 +
                  2.335771970000000e-15 * T4;
    // species 12: CH4
    species[12] = -9.468344590000001e+03 * invT - 1.836246650500000e+01 -
                  7.485149500000000e-02 * logT - 6.695473350000000e-03 * T +
                  9.554763483333333e-07 * T2 - 1.019104458333333e-10 * T3 +
                  5.090761500000000e-15 * T4;
    // species 13: HCO
    species[13] = +4.011918150000000e+03 * invT - 7.026170540000000e+00 -
                  2.772174380000000e+00 * logT - 2.478477630000000e-03 * T +
                  4.140760216666667e-07 * T2 - 4.909681483333334e-11 * T3 +
                  2.667543555000000e-15 * T4;
    // species 14: CH2O
    species[14] = -1.399583230000000e+04 * invT - 1.189563292000000e+01 -
                  1.760690080000000e+00 * logT - 4.600000410000000e-03 * T +
                  7.370980216666666e-07 * T2 - 8.386767666666666e-11 * T3 +
                  4.419278200000001e-15 * T4;
    // species 15: CH3O
    species[15] = +3.781119400000000e+02 * invT + 6.724592660000000e+00 -
                  4.757792380000000e+00 * logT - 3.720712370000000e-03 * T +
                  4.495086266666666e-07 * T2 - 3.650754200000000e-11 * T3 +
                  1.317685490000000e-15 * T4;
    // species 16: CO
    species[16] = -1.415187240000000e+04 * invT - 5.103502110000000e+00 -
                  2.715185610000000e+00 * logT - 1.031263715000000e-03 * T +
                  1.664709618333334e-07 * T2 - 1.917108400000000e-11 * T3 +
                  1.018238580000000e-15 * T4;
    // species 17: CO2
    species[17] = -4.875916600000000e+04 * invT + 1.585822230000000e+00 -
                  3.857460290000000e+00 * logT - 2.207185130000000e-03 * T +
                  3.691356733333334e-07 * T2 - 4.362418233333334e-11 * T3 +
                  2.360420820000000e-15 * T4;
    // species 18: C2H2
    species[18] = +2.593599920000000e+04 * invT + 5.377850850000001e+00 -
                  4.147569640000000e+00 * logT - 2.980833320000000e-03 * T +
                  3.954914200000000e-07 * T2 - 3.895101425000000e-11 * T3 +
                  1.806176065000000e-15 * T4;
    // species 19: C2H3
    species[19] = +3.461287390000000e+04 * invT - 4.770599780000000e+00 -
                  3.016724000000000e+00 * logT - 5.165114600000000e-03 * T +
                  7.801372483333333e-07 * T2 - 8.480274000000000e-11 * T3 +
                  4.313035205000000e-15 * T4;
    // species 20: C2H4
    species[20] = +4.939886140000000e+03 * invT - 8.269258140000002e+00 -
                  2.036111160000000e+00 * logT - 7.322707550000000e-03 * T +
                  1.118463191666667e-06 * T2 - 1.226857691666667e-10 * T3 +
                  6.285303050000000e-15 * T4;
    // species 21: C2H5
    species[21] = +1.285752000000000e+04 * invT - 1.150777788000000e+01 -
                  1.954656420000000e+00 * logT - 8.698636100000001e-03 * T +
                  1.330344446666667e-06 * T2 - 1.460147408333333e-10 * T3 +
                  7.482078800000000e-15 * T4;
    // species 22: C2H6
    species[22] = -1.142639320000000e+04 * invT - 1.404372920000000e+01 -
                  1.071881500000000e+00 * logT - 1.084263385000000e-02 * T +
                  1.670934450000000e-06 * T2 - 1.845100008333333e-10 * T3 +
                  9.500144500000000e-15 * T4;
    // species 23: CH2CHO
    species[23] = -9.695000000000000e+02 * invT + 1.100775780000000e+01 -
                  5.975669900000000e+00 * logT - 4.065295700000000e-03 * T +
                  4.572707500000000e-07 * T2 - 3.391920083333333e-11 * T3 +
                  1.088008550000000e-15 * T4;
    // species 24: aC3H5
    species[24] = +1.748244900000000e+04 * invT + 1.774383770000000e+01 -
                  6.500787700000000e+00 * logT - 7.162365500000000e-03 * T +
                  9.463605333333332e-07 * T2 - 9.234000833333333e-11 * T3 +
                  4.518194349999999e-15 * T4;
    // species 25: C3H6
    species[25] = -9.235703000000000e+02 * invT + 2.004560700000000e+01 -
                  6.732257000000000e+00 * logT - 7.454170000000000e-03 * T +
                  8.249831666666666e-07 * T2 - 6.010018333333334e-11 * T3 +
                  1.883102000000000e-15 * T4;
    // species 26: nC3H7
    species[26] = +7.976223600000000e+03 * invT + 2.322504490000000e+01 -
                  7.709747900000000e+00 * logT - 8.015742500000001e-03 * T +
                  8.786706333333332e-07 * T2 - 6.324029333333334e-11 * T3 +
                  1.943135950000000e-15 * T4;
    // species 27: C2H3CHO
    species[27] = -1.078405400000000e+04 * invT + 1.066998720000000e+01 -
                  5.811186800000000e+00 * logT - 8.557128000000001e-03 * T +
                  1.247236016666667e-06 * T2 - 1.187687416666667e-10 * T3 +
                  4.587342050000000e-15 * T4;
    // species 28: C4H7
    species[28] = +2.095500800000000e+04 * invT + 1.590279150000000e+01 -
                  7.013483500000000e+00 * logT - 1.131727900000000e-02 * T +
                  1.542424500000000e-06 * T2 - 1.400660583333333e-10 * T3 +
                  5.204308500000000e-15 * T4;
    // species 29: C4H81
    species[29] = -2.139723100000000e+03 * invT - 1.348961690000000e+01 -
                  2.053584100000000e+00 * logT - 1.717525350000000e-02 * T +
                  2.647199500000000e-06 * T2 - 2.757471833333334e-10 * T3 +
                  1.268052250000000e-14 * T4;
    // species 30: pC4H9
    species[30] = +4.964405800000000e+03 * invT + 2.657398650000000e+01 -
                  8.682239500000000e+00 * logT - 1.184553550000000e-02 * T +
                  1.265814416666667e-06 * T2 - 5.535594666666666e-11 * T3 -
                  2.742256800000000e-15 * T4;
    // species 31: C5H9
    species[31] = -1.721835900000000e+03 * invT + 4.326452500000000e+01 -
                  1.013864000000000e+01 * logT - 1.135706900000000e-02 * T +
                  1.298507716666667e-06 * T2 - 9.897101666666667e-11 * T3 +
                  3.296622400000000e-15 * T4;
    // species 48: C12H25O2
    species[48] = -3.741180000000000e+04 * invT + 1.375992000000000e+02 -
                  2.847820000000000e+01 * logT - 2.687695000000000e-02 * T +
                  2.803100000000000e-06 * T2 - 2.094725000000000e-10 * T3 +
                  7.360400000000000e-15 * T4;
    // species 49: C12OOH
    species[49] = -3.111920000000000e+04 * invT + 1.380569000000000e+02 -
                  2.920190000000000e+01 * logT - 2.579585000000000e-02 * T +
                  2.622116666666667e-06 * T2 - 1.919216666666667e-10 * T3 +
                  6.632000000000000e-15 * T4;
    // species 50: O2C12H24OOH
    species[50] = -5.126750000000000e+04 * invT + 1.728407000000000e+02 -
                  3.509070000000000e+01 * logT - 2.552950000000000e-02 * T +
                  2.572416666666666e-06 * T2 - 1.871891666666667e-10 * T3 +
                  6.445050000000000e-15 * T4;
    // species 51: OC12H23OOH
    species[51] = -7.182580000000000e+04 * invT + 1.014393000000000e+02 -
                  2.367310000000000e+01 * logT - 3.081960000000000e-02 * T +
                  3.497266666666667e-06 * T2 - 2.776383333333333e-10 * T3 +
                  1.017950000000000e-14 * T4;
    // species 52: N2
    species[52] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                  2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                  9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                  3.376675500000000e-16 * T4;
  }

  // species with midpoint at T=1385 kelvin
  if (T < 1385) {
    // species 45: SXC12H25
    species[45] = -1.676605390000000e+04 * invT - 4.972006039000000e+01 +
                  1.367870890000000e+00 * logT - 6.867767399999999e-02 * T +
                  1.373460263333333e-05 * T2 - 1.970179683333333e-09 * T3 +
                  1.237179660000000e-13 * T4;
    // species 46: S3XC12H25
    species[46] = -1.676605390000000e+04 * invT - 4.972006039000000e+01 +
                  1.367870890000000e+00 * logT - 6.867767399999999e-02 * T +
                  1.373460263333333e-05 * T2 - 1.970179683333333e-09 * T3 +
                  1.237179660000000e-13 * T4;
  } else {
    // species 45: SXC12H25
    species[45] = -3.121449880000000e+04 * invT + 2.037747598000000e+02 -
                  3.796882680000000e+01 * logT - 2.693597320000000e-02 * T +
                  3.036187716666667e-06 * T2 - 2.339787525000000e-10 * T3 +
                  8.105421000000000e-15 * T4;
    // species 46: S3XC12H25
    species[46] = -3.121449880000000e+04 * invT + 2.037747598000000e+02 -
                  3.796882680000000e+01 * logT - 2.693597320000000e-02 * T +
                  3.036187716666667e-06 * T2 - 2.339787525000000e-10 * T3 +
                  8.105421000000000e-15 * T4;
  }

  // species with midpoint at T=1390 kelvin
  if (T < 1390) {
    // species 33: PXC5H11
    species[33] = +4.716114600000000e+03 * invT - 2.867142819190000e+01 -
                  5.243840810000000e-02 * logT - 2.803984790000000e-02 * T +
                  5.525763383333334e-06 * T2 - 8.146114841666667e-10 * T3 +
                  5.700483000000000e-14 * T4;
    // species 35: PXC6H13
    species[35] = +1.832803930000000e+03 * invT - 3.181238076500000e+01 +
                  2.048714650000000e-01 * logT - 3.419006360000000e-02 * T +
                  6.907465199999999e-06 * T2 - 1.051298350000000e-09 * T3 +
                  7.656002899999999e-14 * T4;
    // species 37: PXC7H15
    species[37] = -1.045902230000000e+03 * invT - 3.515597150599999e+01 +
                  4.995704060000000e-01 * logT - 4.044132335000000e-02 * T +
                  8.342212566666667e-06 * T2 - 1.304577566666667e-09 * T3 +
                  9.830811350000000e-14 * T4;
    // species 39: PXC8H17
    species[39] = -3.926895110000000e+03 * invT - 3.838582253800000e+01 +
                  7.727594380000000e-01 * logT - 4.662748525000000e-02 * T +
                  9.740787416666667e-06 * T2 - 1.546418450000000e-09 * T3 +
                  1.185637415000000e-13 * T4;
    // species 41: PXC9H19
    species[41] = -6.808185120000000e+03 * invT - 4.339577212000000e+01 +
                  1.043872920000000e+00 * logT - 5.280864150000000e-02 * T +
                  1.113666618333333e-05 * T2 - 1.787384716666667e-09 * T3 +
                  1.387021375000000e-13 * T4;
    // species 43: PXC10H21
    species[43] = -9.689675499999999e+03 * invT - 4.481462868000000e+01 +
                  1.313583480000000e+00 * logT - 5.898640650000000e-02 * T +
                  1.253071798333333e-05 * T2 - 2.027759216666667e-09 * T3 +
                  1.587614260000000e-13 * T4;
    // species 44: PXC12H25
    species[44] = -1.545304350000000e+04 * invT - 5.122052951000000e+01 +
                  1.850287410000000e+00 * logT - 7.133535400000000e-02 * T +
                  1.531527591666667e-05 * T2 - 2.507361600000000e-09 * T3 +
                  1.987271500000000e-13 * T4;
  } else {
    // species 33: PXC5H11
    species[33] = -9.807123070000000e+02 * invT + 6.978067390000000e+01 -
                  1.529774460000000e+01 * logT - 1.198676550000000e-02 * T +
                  1.363988246666667e-06 * T2 - 1.057358966666667e-10 * T3 +
                  3.677045275000000e-15 * T4;
    // species 35: PXC6H13
    species[35] = -5.092990410000000e+03 * invT + 8.898764130000001e+01 -
                  1.853854700000000e+01 * logT - 1.415539810000000e-02 * T +
                  1.608845410000000e-06 * T2 - 1.246229875000000e-10 * T3 +
                  4.331680320000000e-15 * T4;
    // species 37: PXC7H15
    species[37] = -9.209382210000000e+03 * invT + 1.082895020000000e+02 -
                  2.179407090000000e+01 * logT - 1.631401215000000e-02 * T +
                  1.852304066666667e-06 * T2 - 1.433892900000000e-10 * T3 +
                  4.981834995000000e-15 * T4;
    // species 39: PXC8H17
    species[39] = -1.333005350000000e+04 * invT + 1.276084196000000e+02 -
                  2.505103560000000e+01 * logT - 1.847400810000000e-02 * T +
                  2.096087733333333e-06 * T2 - 1.621903408333333e-10 * T3 +
                  5.633444900000000e-15 * T4;
    // species 41: PXC9H19
    species[41] = -1.745160300000000e+04 * invT + 1.451476484000000e+02 -
                  2.830975140000000e+01 * logT - 2.063286720000000e-02 * T +
                  2.339721483333333e-06 * T2 - 1.809790591666667e-10 * T3 +
                  6.284615349999999e-15 * T4;
    // species 43: PXC10H21
    species[43] = -2.157378320000000e+04 * invT + 1.662787020000000e+02 -
                  3.156971600000000e+01 * logT - 2.279092015000000e-02 * T +
                  2.583249416666667e-06 * T2 - 1.997591108333333e-10 * T3 +
                  6.935477950000000e-15 * T4;
    // species 44: PXC12H25
    species[44] = -2.981943750000000e+04 * invT + 2.049749225000000e+02 -
                  3.809218850000000e+01 * logT - 2.710539240000000e-02 * T +
                  3.070091950000000e-06 * T2 - 2.373018108333333e-10 * T3 +
                  8.236587400000000e-15 * T4;
  }

  // species with midpoint at T=1391 kelvin
  if (T < 1391) {
    // species 0: NC12H26
    species[0] = -4.006542530000000e+04 * invT - 5.272127854000000e+01 +
                 2.621815940000000e+00 * logT - 7.361885550000000e-02 * T +
                 1.573283785000000e-05 * T2 - 2.562010566666667e-09 * T3 +
                 2.018011150000000e-13 * T4;
    // species 47: C12H24
    species[47] = -2.463452990000000e+04 * invT - 5.587931381000000e+01 +
                  2.963426810000000e+00 * logT - 7.199618000000001e-02 * T +
                  1.602306691666667e-05 * T2 - 2.751453941666667e-09 * T3 +
                  2.311990950000000e-13 * T4;
  } else {
    // species 0: NC12H26
    species[0] = -5.488434650000000e+04 * invT + 2.111804257000000e+02 -
                 3.850950370000000e+01 * logT - 2.817750240000000e-02 * T +
                 3.191553333333333e-06 * T2 - 2.466873850000000e-10 * T3 +
                 8.562207500000000e-15 * T4;
    // species 47: C12H24
    species[47] = -3.894059620000000e+04 * invT + 2.022928741000000e+02 -
                  3.740021110000000e+01 * logT - 2.631153765000000e-02 * T +
                  2.977071983333333e-06 * T2 - 2.299582191666667e-10 * T3 +
                  7.978124949999999e-15 * T4;
  }

  // species with midpoint at T=1392 kelvin
  if (T < 1392) {
    // species 32: C5H10
    species[32] = -4.465466660000000e+03 * invT - 3.333621381000000e+01 +
                  1.062234810000000e+00 * logT - 2.871091470000000e-02 * T +
                  6.241448166666667e-06 * T2 - 1.061374908333333e-09 * T3 +
                  8.980489449999999e-14 * T4;
    // species 34: C6H12
    species[34] = -7.343686170000000e+03 * invT - 3.666482115000000e+01 +
                  1.352752050000000e+00 * logT - 3.493277130000000e-02 * T +
                  7.656800366666667e-06 * T2 - 1.308061191666667e-09 * T3 +
                  1.106480875000000e-13 * T4;
    // species 36: C7H14
    species[36] = -1.021686010000000e+04 * invT - 4.018400869000000e+01 +
                  1.677205490000000e+00 * logT - 4.123058005000000e-02 * T +
                  9.108401800000000e-06 * T2 - 1.565519191666667e-09 * T3 +
                  1.328689915000000e-13 * T4;
    // species 38: C8H16
    species[38] = -1.310745590000000e+04 * invT - 4.308016725000000e+01 +
                  1.892269150000000e+00 * logT - 4.730331785000000e-02 * T +
                  1.045642535000000e-05 * T2 - 1.792985908333333e-09 * T3 +
                  1.513593415000000e-13 * T4;
    // species 40: C9H18
    species[40] = -1.598908470000000e+04 * invT - 4.628559543000000e+01 +
                  2.161082630000000e+00 * logT - 5.347914850000000e-02 * T +
                  1.184955406666667e-05 * T2 - 2.033092308333333e-09 * T3 +
                  1.713857735000000e-13 * T4;
    // species 42: C10H20
    species[42] = -1.887083650000000e+04 * invT - 4.948615518000000e+01 +
                  2.429016880000000e+00 * logT - 5.965279900000000e-02 * T +
                  1.324148375000000e-05 * T2 - 2.272804966666667e-09 * T3 +
                  1.913591865000000e-13 * T4;
  } else {
    // species 32: C5H10
    species[32] = -1.008982050000000e+04 * invT + 6.695354750000000e+01 -
                  1.458515390000000e+01 * logT - 1.120362355000000e-02 * T +
                  1.272246708333333e-06 * T2 - 9.849080500000001e-11 * T3 +
                  3.421925695000000e-15 * T4;
    // species 34: C6H12
    species[34] = -1.420628600000000e+04 * invT + 8.621563800000001e+01 -
                  1.783375290000000e+01 * logT - 1.336888290000000e-02 * T +
                  1.516727955000000e-06 * T2 - 1.173498066666667e-10 * T3 +
                  4.075621220000000e-15 * T4;
    // species 36: C7H14
    species[36] = -1.832600650000000e+04 * invT + 1.055289147000000e+02 -
                  2.108980390000000e+01 * logT - 1.553039390000000e-02 * T +
                  1.760746550000000e-06 * T2 - 1.361714833333333e-10 * T3 +
                  4.727991095000000e-15 * T4;
    // species 38: C8H16
    species[38] = -2.244856740000000e+04 * invT + 1.248917285000000e+02 -
                  2.435401250000000e+01 * logT - 1.768332310000000e-02 * T +
                  2.003473133333334e-06 * T2 - 1.548792108333333e-10 * T3 +
                  5.376113100000000e-15 * T4;
    // species 40: C9H18
    species[40] = -2.657090610000000e+04 * invT + 1.442328406000000e+02 -
                  2.761421760000000e+01 * logT - 1.984126435000000e-02 * T +
                  2.246990766666667e-06 * T2 - 1.736587100000000e-10 * T3 +
                  6.026964700000000e-15 * T4;
    // species 42: C10H20
    species[42] = -3.069373070000000e+04 * invT + 1.635805623000000e+02 -
                  3.087539030000000e+01 * logT - 2.199857630000000e-02 * T +
                  2.490425500000000e-06 * T2 - 1.924313983333333e-10 * T3 +
                  6.677573850000000e-15 * T4;
  }
}

#endif
