#include"MemoryMonitor.h"
#include<cuda_runtime.h>
#include<stdlib.h>
#include<string.h>

#include"checkError.h"

void * MemoryMonitor::cpuMallocMemory(int size){

	void* p=NULL;
	p = malloc(size);
	return p;
}


void MemoryMonitor::gpuMallocMemory(void**devPtr, int size)
{
	if(*devPtr != NULL)
	{
		checkCudaErrors(cudaFree(*devPtr));
	}
	checkCudaErrors(cudaMalloc(devPtr, size));
}


void MemoryMonitor::gpuMemoryMemset(void *dev_data, int size)
{
	//cudaMemset只能清零，赋为其他值有很大差异，和memset类似，请看其具体用法
	checkCudaErrors(cudaMemset(dev_data, 0 , size));
}

void MemoryMonitor::cpuMemoryMemset(void * host_data,int size)
{
	memset(host_data, 0, size);
}

void MemoryMonitor::freeCpuMemory(void *ptr)
{
	free(ptr);
}


void MemoryMonitor::freeGpuMemory(void *ptr)
{
	checkCudaErrors(cudaFree(ptr));
}
