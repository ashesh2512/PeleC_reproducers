#include "hip/hip_runtime.h"

__global__ void reverse(int *d, int n)
{
#if USE_VOLATILE
  __shared__ volatile int s[64];
#else
  __shared__ int s[64];
#endif
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;
  int a[n], d[n];

  for (int i = 0; i < n; i++) {
    a[i] = i;
    d[i] = 0;
  }

  int *d_d;
  hipMalloc(&d_d, n * sizeof(int)); 

  hipMemcpy(d_d, a, n*sizeof(int), hipMemcpyHostToDevice);
  reverse<<<1,n>>>(d_d, n);

  return 0;
}
