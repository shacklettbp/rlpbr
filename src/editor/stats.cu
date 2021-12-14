#include "stats.hpp"

#include <iostream>
#include <vector>

using namespace std;

namespace RLpbr {

static __host__ inline cudaError_t checkCuda(cudaError_t res, const char *msg,
                                             bool fatal = true) noexcept
{
    if (res != cudaSuccess) {
        cerr << msg << ": " << cudaGetErrorString(res) << endl;
        if (fatal) {
            abort();
        }
    }

    return res;
}

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)
#define REQ_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr))
#define CHK_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr), false)

__device__ inline double3 warpReduceSum(double3 cur_sum) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        cur_sum.x += __shfl_down_sync(0xffffffff, cur_sum.x, offset);
        cur_sum.y += __shfl_down_sync(0xffffffff, cur_sum.y, offset);
        cur_sum.z += __shfl_down_sync(0xffffffff, cur_sum.z, offset);
    }

    return cur_sum;
}

template <unsigned int blockSize>
inline __device__ double3 blockReduceSum(double3 thread_sum)
{
    static __shared__ double3 shared[blockSize / 32];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    double3 warp_sum = warpReduceSum(thread_sum);

    if (lane == 0) {
        shared[wid] = warp_sum;
    }

    __syncthreads();

    double3 block_sum = (threadIdx.x < blockDim.x / warpSize) ?
        shared[lane] : make_double3(0.0, 0.0, 0.0);

    if (wid == 0) {
        block_sum = warpReduceSum(block_sum);
    }

    return block_sum;
}

template <unsigned int blockSize, typename Fn>
__global__ void sum(double *g_odata, int64_t n,
                    Fn get_data) {

    double3 thread_sum = make_double3(0.0, 0.0, 0.0);
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        thread_sum.x += get_data(i, 0);
        thread_sum.y += get_data(i, 1);
        thread_sum.z += get_data(i, 2);
    }

    double3 block_sum = blockReduceSum<blockSize>(thread_sum);

    if (threadIdx.x == 0) {
        g_odata[3 * blockIdx.x] = block_sum.x;
        g_odata[3 * blockIdx.x + 1] = block_sum.y;
        g_odata[3 * blockIdx.x + 2] = block_sum.z;
    }
}

constexpr int mv_block_size = 512;

__host__ MeanVarContext getMeanVarContext(uint32_t batch_size,
                                          uint32_t res_x,
                                          uint32_t res_y)
{
    int64_t num_pixels = batch_size * res_x * res_y;

    int num_blocks = (num_pixels + mv_block_size - 1) / mv_block_size;
    num_blocks = min(num_blocks, 1024);

    // Allocate scratch space with space for 1 additional block,
    // which contains mean results while var kernel is running
    double *scratch;
    REQ_CUDA(cudaMalloc(&scratch, sizeof(double) * (num_blocks + 1) * 3));

    cudaStream_t strm;
    REQ_CUDA(cudaStreamCreate(&strm));

    return {
        num_pixels,
        num_blocks,
        scratch,
        strm,
    };
}

__host__ pair<array<float, 3>, array<float, 3>> computeMeanAndVar(
    half *batch_raw, const MeanVarContext &ctx)
{
    sum<mv_block_size><<<ctx.numBlocks, mv_block_size, 0, ctx.strm>>>(
        ctx.scratch, ctx.numPixels,
        [batch_raw] __device__ (int64_t base, int channel) {
            return double(batch_raw[4 * base + channel]);
        });

    sum<mv_block_size><<<1, ctx.numBlocks, 0, ctx.strm>>>(
        ctx.scratch, ctx.numBlocks,
        [scratch=ctx.scratch] __device__ (int64_t base, int channel) {
            return scratch[3 * base + channel];
        });

    sum<mv_block_size><<<ctx.numBlocks, mv_block_size, 0, ctx.strm>>>(
        ctx.scratch + 3, ctx.numPixels,
        [batch_raw, scratch=ctx.scratch, num_pixels=ctx.numPixels] __device__ (
                int64_t base, int channel) {
            double input = double(batch_raw[4 * base + channel]);
            double mean = scratch[channel] / double(num_pixels);

            double diff = input - mean;

            return diff * diff;
        });

    sum<mv_block_size><<<1, ctx.numBlocks, 0, ctx.strm>>>(
        ctx.scratch + 3, ctx.numBlocks,
        [scratch=(ctx.scratch + 3)] __device__ (int64_t base, int channel) {
            return scratch[3 * base + channel];
        });

    array<double, 6> result;
    cudaMemcpyAsync(result.data(), ctx.scratch, 6 * sizeof(double),
                    cudaMemcpyDeviceToHost, ctx.strm);

    REQ_CUDA(cudaStreamSynchronize(ctx.strm));

    array<float, 3> mean {
        float(result[0] / ctx.numPixels),
        float(result[1] / ctx.numPixels),
        float(result[2] / ctx.numPixels),
    };

    array<float, 3> var {
        float(result[3] / ctx.numPixels),
        float(result[4] / ctx.numPixels),
        float(result[5] / ctx.numPixels),
    };

    return {
        mean,
        var,
    };

#if 0
    vector<half> data(ctx.numPixels * 4);
    REQ_CUDA(cudaMemcpy(data.data(), batch_raw, data.size() * sizeof(half),
                        cudaMemcpyDeviceToHost));

    array<double, 3> mean2 { 0, 0, 0 };
    for (int64_t i = 0; i < (int64_t)ctx.numPixels; i++) {
        for (int c = 0; c < 3; c++) {
            mean2[c] += double(data[4 * i + c]);
        }
    }

    mean2[0] /= ctx.numPixels;
    mean2[1] /= ctx.numPixels;
    mean2[2] /= ctx.numPixels;

    array<double, 3> var2 { 0, 0, 0 };
    for (int64_t i = 0; i < (int64_t)ctx.numPixels; i++) {
        for (int c = 0; c < 3; c++) {
            double input = double(data[4 * i + c]);
            double diff = input - mean2[c];
            var2[c] += diff * diff;
        }
    }

    var2[0] /= ctx.numPixels;
    var2[1] /= ctx.numPixels;
    var2[2] /= ctx.numPixels;

    printf("%f %f %f\n%f %f %f\n", mean[0], mean[1], mean[2], var[0], var[1],
           var[2]);
    printf("%f %f %f\n%f %f %f\n", mean2[0], mean2[1], mean2[2], var2[0], var2[1],
           var2[2]);
#endif
}

}
