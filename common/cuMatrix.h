/*
 * cuMatrix.h
 *
 *  Created on: Nov 19, 2015
 *      Author: tdx
 */

#ifndef CUMATRIX_H_
#define CUMATRIX_H_

#include"MemoryMonitor.h"
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<cuda_runtime.h>
#include"checkError.h"

/*row major*/
template<typename T>
class cuMatrix
{
public:
	cuMatrix(T * _data, int _r, int _c, int _ch, bool _isGpuData = false):rows(_r),cols(_c),channels(_ch),hostData(NULL),devData(NULL){
        if( _isGpuData == false){
            /*allocate host memory*/
            mallocHostMemory();
            /*deep copy*/
            memcpy(hostData,_data,rows * cols *sizeof(*hostData) * channels);
        }
        else{
            mallocDeviceMemory();
            MemoryMonitor::instanceObject()->gpu2gpu(devData, _data, rows * cols * sizeof(*devData) * channels );
        }
    }

    /*constructed function with rows and cols*/
    cuMatrix(int _r, int _c, int _ch):rows(_r),cols(_c), channels(_ch),hostData(NULL), devData(NULL){}

    /*destructor*/
    ~cuMatrix()
    {
        if(hostData != NULL)
        {
            MemoryMonitor::instanceObject()->freeCpuMemory(hostData);
        }
        if(devData != NULL)
        {
            MemoryMonitor::instanceObject()->freeGpuMemory(devData);
        }
    }

    /*set value*/
    void setValue(int i, int j, int k, T v)
    {
        mallocHostMemory();
        hostData[(i* cols + j) + rows * cols * k] = v;
    }

    /*get value*/
    T getValue(int i, int j, int k)
    {
        return hostData[(i * cols+ j) + rows * cols * k];
    }

    /*copy the host data to device*/
    void toGpu()
    {
        mallocHostMemory();
        mallocDeviceMemory();
        checkCudaErrors(cudaMemcpy(devData, hostData, rows * cols * sizeof(*devData) * channels,cudaMemcpyHostToDevice));
    }

    /*copy the data to host*/
    void toCpu()
    {
        mallocDeviceMemory();
        mallocHostMemory();
        checkCudaErrors(cudaMemcpy(hostData, devData, sizeof(*hostData)*rows* cols * channels,cudaMemcpyDeviceToHost));
    }

    /*get the number of value*/
    int getLength()
    {
        return rows * cols * channels;
    }

    int getArea()
    {
        return rows * cols;
    }

    T* &getHostData()
    {
        return hostData;
    }

    /*get device data*/
    T* &getDeviceData()
    {
        return devData;
    }


public:
    int rows;
    int cols;
    int channels;

private:
    /*host data*/
    T *hostData;
    /*device data*/
    T *devData;

private:
    /*allocate host memory*/
    void mallocHostMemory()
    {
        if(hostData==NULL)
        {
            hostData=(T*)MemoryMonitor::instanceObject()->cpuMallocMemory(rows * cols * sizeof(*hostData) * channels);
            if(!hostData)
            {
                printf("cuMatrix:cuMatrix host Memory allocation Failed\n");
                exit(0);
            }

            /*init allocation memory*/
            memset(hostData,0,rows * cols * channels * sizeof(*hostData));
        }
    }


    /*allocate device memory*/
    void mallocDeviceMemory()
    {
        if(devData == NULL)
        {
            /*malloc device data*/
            MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&devData, rows * cols * channels * sizeof(*devData));
            checkCudaErrors(cudaMemset(devData, 0, rows * cols * channels * sizeof(*devData)));
        }
    }
};



#endif /* CUMATRIX_H_ */
