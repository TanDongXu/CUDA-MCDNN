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
    void cpu2cpu(void* host_data2, void* host_data1, int size);
    void cpu2Gpu(void* dev_data, void* host_data, int size);
    void gpu2cpu(void* host_data, void* dev_data, int size);
    void gpu2gpu(void* dev_data2, void* dev_data1, int size);
};



#endif /* MEMORYMONITOR_H_ */
