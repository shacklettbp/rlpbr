#include "utils.hpp"

#include <optix_stubs.h>

#include <iostream>

using namespace std;

namespace RLpbr {
namespace optix {

void printOptixError(OptixResult res, const char *msg)
{
    cerr << msg << ": " << optixGetErrorString(res) << endl;
}

void printCudaError(cudaError_t res, const char *msg)
{
    cerr << msg << ": " << cudaGetErrorString(res) << endl;
}

void printNVRTCError(nvrtcResult res, const char *msg)
{
    cerr << msg << ": " << nvrtcGetErrorString(res) << endl;
}


}
}
