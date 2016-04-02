#include"MemoryMonitor.h"
#include<cuda_runtime.h>
#include<stdlib.h>
#include<string.h>
#include<cstring>
#include"checkError.h"


/*allocate cpu memory*/
void * MemoryMonitor::cpuMallocMemory(int size){
	void* p=NULL;
	p = malloc(size);
	return p;
}

void MemoryMonitor::cpu2cpu(void* host_data2, void* host_data1, int size)
{
	memcpy(host_data2, host_data1, size);
}

void MemoryMonitor::cpu2Gpu(void* dev_data, void* host_data, int size)
{
    checkCudaErrors(cudaMemcpy(dev_data, host_data, size, cudaMemcpyHostToDevice));
}

void MemoryMonitor::gpu2cpu(void* host_data, void* dev_data, int size)
{
    checkCudaErrors(cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost));
}

void MemoryMonitor::gpu2gpu(void* dev_data2, void* dev_data1, int size)
{
    checkCudaErrors(cudaMemcpy(dev_data2, dev_data1, size, cudaMemcpyDeviceToDevice));
}

/*allocate GPU memory*/
void MemoryMonitor::gpuMallocMemory(void**devPtr, int size)
{
    if(*devPtr != NULL)
    {
        checkCudaErrors(cudaFree(*devPtr));
    }
    checkCudaErrors(cudaMalloc(devPtr, size));
}

/*GPU memory set zero*/
void MemoryMonitor::gpuMemoryMemset(void *dev_data, int size)
{
    checkCudaErrors(cudaMemset(dev_data, 0 , size));
}

/*cpu memory set zero*/
void MemoryMonitor::cpuMemoryMemset(void * host_data,int size)
{
    memset(host_data, 0, size);
}

/*free cpu memory*/
void MemoryMonitor::freeCpuMemory(void *ptr)
{
    free(ptr);
}

/*free Device memory*/
void MemoryMonitor::freeGpuMemory(void *ptr)
{
    checkCudaErrors(cudaFree(ptr));
}
