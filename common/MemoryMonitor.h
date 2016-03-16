/*
 * MemoryMonitor.h
 *
 *  Created on: Nov 19, 2015
 *      Author: tdx
 */

#ifndef MEMORYMONITOR_H_
#define MEMORYMONITOR_H_

#include<cuda_runtime.h>

class MemoryMonitor
{
public:
	static MemoryMonitor *instanceObject(){
		static MemoryMonitor *monitor=new MemoryMonitor();
		return monitor;
	}

	void *cpuMallocMemory(int size);
	void gpuMallocMemory(void** devPtr ,int size);

	void freeCpuMemory(void* ptr);
	void freeGpuMemory(void*ptr);
	void gpuMemoryMemset(void* dev_data, int size);
	void cpuMemoryMemset(void* host_data, int size);

};



#endif /* MEMORYMONITOR_H_ */
