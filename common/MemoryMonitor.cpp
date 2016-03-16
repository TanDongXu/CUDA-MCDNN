#include"MemoryMonitor.h"
#include<cuda_runtime.h>
#include<stdlib.h>
#include<string.h>
#include"checkError.h"


/*allocate cpu memory*/
void * MemoryMonitor::cpuMallocMemory(int size){

	void* p=NULL;
	p = malloc(size);
	return p;
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
