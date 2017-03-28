#include<iostream>
#include<cuda_runtime.h>
#include"test_layer.h"
#include<glog/logging.h>
#include"../common/checkError.h"
#include"../common/MemoryMonitor.h"

using namespace std;

/*judge array equal or not*/

void CHECK_HOST_MATRIX_EQ(float*A, int sizeA, float*B, int sizeB)
{
    string s ="NOT_EQ";
    if(sizeA != sizeB) FatalError(s);
    cout<<sizeA<<endl;
    for(int i=0; i< sizeA/sizeof(float); i++)
    {
        CHECK_EQ(A[i],B[i]);
    }
}

/*judge cpu data and GPU data equal or not*/
void CHECK_DEV_HOST_MATRIX_EQ(float* A, int sizeA, float* B, int sizeB)
{
    float* tmpB;
    tmpB = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(sizeB);
    checkCudaErrors(cudaMemcpy(tmpB, B, sizeB, cudaMemcpyDeviceToHost));
    CHECK_HOST_MATRIX_EQ(A, sizeA, tmpB, sizeB);
}

/*printf hostdata，channels,height,width*/
//template<class T>
void printf_HostParameter(int number, int channels, int height,int width, float*A)
{
    for(int n=0; n<number; n++)
    {
        for(int c=0; c<channels; c++)
        {
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<width; w++)
                {
                    cout<<A[h * width + w + height * width * c + height * width * channels * n]<<" ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
    }
    cout<<"over"<<endl;
    for(;;){}
}

/*printf devData*/
void printf_DevParameter(int number, int channels, int height,int width, float*A)
{
    float*tmpA;
    tmpA = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(number*channels*height*width*sizeof(float));
    checkCudaErrors(cudaMemcpy(tmpA, A,number*channels*height*width*sizeof(float), cudaMemcpyDeviceToHost));

    for(int n=0; n<number; n++)
    {
        for(int c=0; c<channels; c++)
        {
            for(int h=0; h<height; h++)
            {
                for(int w=0; w<width; w++)
                {
                    cout<<tmpA[h * width + w + height * width * c + height * width * channels * n]<<" ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
    }
    cout<<"sdfhdgggfhfhdfhdfgh"<<endl;
    for(;;){}
}

//the second parameter must be reference(引用)
void copy_DeviceToHost(float*devData, float*&hostData, int number, int channels, int height, int width)
{
    hostData = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
    checkCudaErrors(cudaMemcpy(hostData, devData, number * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost));
}

void copy_HostToDevice(float*hostData, float*&devData, int number, int channels, int height, int width)
{
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&devData, number * channels * height * width * sizeof(float));
    checkCudaErrors(cudaMemcpy(devData, hostData, number * channels * height * width * sizeof(float), cudaMemcpyHostToDevice));
}

void printfLayersParameter(LayersBase* layer)
{
    cout<<"name: "<<layer->_name<<endl;
    cout<<"Input: "<<layer->_inputName<<endl;
    cout<<"number: "<<layer->number<<endl;
    cout<<"channels: "<<layer->channels<<endl;
    cout<<"height: "<<layer->height<<endl;
    cout<<"width: "<<layer->width<<endl;
    cout<<"InputAmount: "<<layer->inputAmount<<endl;
    cout<<"InputImageDim: "<<layer->inputImageDim<<endl;
    printf_DevParameter(layer->number, layer->channels, layer->height, layer->width, layer->dstData);

    for(;;){}
}
