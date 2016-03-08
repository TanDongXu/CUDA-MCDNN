/*
 * utility.h
 *
 *  Created on: Nov 24, 2015
 *      Author: tdx
 */

#ifndef UTILITY_H_
#define UTILITY_H_

#include<cuda_runtime.h>
#include<iostream>

#include"checkError.h"

#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

/*the ways of initial weight*/
#define RANDOM 1
#define READ_FROM_FILE 2



/*展示GPU信息*/
static void showDevices()
{
	int totalDevices;
	cudaGetDeviceCount(&totalDevices);
	std::cout<<"There are "<<totalDevices<<" CUDA capable devices on your machine: "<<std::endl;

	for(int i=0; i< totalDevices; i++)
	{
		struct cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, i));
		/*multiProcessorCount设备上多处理器的数量*/
		printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",i, prop.multiProcessorCount, prop.major, prop.minor,
                (float)prop.clockRate*1e-3,
                (int)(prop.totalGlobalMem/(1024*1024)),
                (float)prop.memoryClockRate*1e-3,
                prop.ECCEnabled,
                prop.multiGpuBoardGroupID);
	}

	printf("\n");

}


#endif /* UTILITY_H_ */
