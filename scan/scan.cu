#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


// upsweep and downsweep parallel prefix sum algorithm:https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// upsweep phase conform the original array of :
// [x0][\sum{x0..x1}][x2][\sum{x0..x3}]...[xN-1][\sum{x0..xN}]
__global__ void upsweep(const int twoPowD, const int N, int* input, int* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int twoPowDPlus = twoPowD << 1;
    int k = twoPowDPlus * idx;
    if(k < N){
        output[k + twoPowDPlus - 1] = input[k + twoPowD - 1] + output[k + twoPowDPlus - 1];
    }
}

__global__ void downsweep(const int twoPowD, const int N, int* input, int* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int twoPowDPlus = twoPowD << 1;
    int k = twoPowDPlus * idx;
    if(k < N){
        int temp = input[k + twoPowD - 1];
        output[k + twoPowD - 1] = input[k + twoPowDPlus -1];
        output[k + twoPowDPlus - 1] = temp + input[k + twoPowDPlus - 1];
    }
}

// 

// exclusiveScan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusiveScan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    // copy input to result
    cudaMemcpy(result, input, N * sizeof(int), cudaMemcpyDeviceToDevice);
    // upsweep phase
    int nextPow2Length = nextPow2(N);
    for(int twoPowD = 1; twoPowD <= nextPow2Length / 2; twoPowD <<= 1){       
        int threadPerBlock = std::min(nextPow2Length / (twoPowD << 1), THREADS_PER_BLOCK);
        int numBlocks = (nextPow2Length / (twoPowD << 1) + threadPerBlock - 1) / threadPerBlock;
        upsweep<<<numBlocks, threadPerBlock>>>(twoPowD, nextPow2Length, result, result);
        cudaDeviceSynchronize(); // wait all threads to finish the sum process
    }
    // set the last element to 0
    cudaMemset(&result[nextPow2Length-1], 0, sizeof(int));
    for(int twoPowD = nextPow2Length / 2; twoPowD >= 1; twoPowD >>= 1){
        int threadPerBlock = std::min(nextPow2Length / (twoPowD << 1), THREADS_PER_BLOCK);
        int numBlocks = (nextPow2Length / (twoPowD << 1) + threadPerBlock - 1) / threadPerBlock;
        downsweep<<<numBlocks, threadPerBlock>>>(twoPowD, nextPow2Length, result, result);
        cudaDeviceSynchronize(); // wait all threads to finish the sum process
    }
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusiveScan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusiveScan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusiveScan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int roundedLength = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * roundedLength);
    cudaMalloc((void **)&device_input, sizeof(int) * roundedLength);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusiveScan from findRepeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusiveScan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


__global__ void repete(int length, int* array, int* result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length - 1 && array[idx] == array[idx+1]){
        result[idx] = 1;
    } else if(idx == length - 1){
        result[idx] = 0;
    }
}

__global__ void gather(int length, int* totalNum, int* scanned, int* result){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length - 1){
        if(scanned[idx] != scanned[idx+1]){
            result[scanned[idx]] = idx;
        }
    } else if(idx == length - 1){
        *totalNum = scanned[length-1];
    }
}

// findRepeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int findRepeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusiveScan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusiveScan function with them. However, your implementation
    // must ensure that the results of findRepeats are correct given
    // the actual array length.
    int nextPow2Length = nextPow2(length);
    int* result, *scanned;
    cudaMalloc(&result, nextPow2Length * sizeof(int));
    int threadPerBlock = std::min(nextPow2Length, THREADS_PER_BLOCK);
    int numBlocks = (nextPow2Length + threadPerBlock - 1) / threadPerBlock;
    repete<<<numBlocks, threadPerBlock>>>(length, device_input, result);

    cudaMalloc(&scanned, nextPow2Length * sizeof(int));
    exclusiveScan(result, length, scanned);
    cudaFree(result);

    int* totalNum;
    cudaMalloc(&totalNum, sizeof(int));
    gather<<<numBlocks, threadPerBlock>>>(length, totalNum, scanned, device_output);
    cudaFree(scanned);

    int outputLength;
    cudaMemcpy(&outputLength, totalNum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(totalNum);
    
    return outputLength; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around findRepeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *outputLength) {

    int *device_input;
    int *device_output;
    int roundedLength = nextPow2(length);
    
    cudaMalloc((void **)&device_input, roundedLength * sizeof(int));
    cudaMalloc((void **)&device_output, roundedLength * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = findRepeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *outputLength = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
