/*
* cuDNN_netWork.h
*
*  Created on: Dec 8, 2015
*      Author: tdx
*/

#ifndef CUDNN_NETWORK_H_
#define CUDNN_NETWORK_H_

#include"./layers/convLayer.h"
#include"./layers/hiddenLayer.h"
#include<cuda_runtime.h>
#include<cudnn.h>
#include<cublas.h>
#include<cublas_v2.h>
#include"./common/checkError.h"

void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                   cudnnTensorFormat_t& tensorFormat, 
                   cudnnDataType_t& dataType,
                   int n,
                   int c,
                   int h,
                   int w);


void matrixMulti(cublasHandle_t cublasHandle, 
                 int m, 
                 int n, 
                 int batchSize, 
                 float alpha,
                 const float*A, 
                 const float*x, 
                 float beta, 
                 float *y);



template <class T>
class cuDNN_netWork
{
    public:
    cudnnDataType_t& GetDataType()
    {
        return  dataType;
    }

    cudnnTensorFormat_t& GetTensorFormat()
    {
        return tensorFormat;
    }

    cudnnHandle_t& GetcudnnHandle()
    {
        return cudnnHandle;
    }


    cublasHandle_t& GetcublasHandle()
    {
        return cublasHandle;
    }


    int& GetconvAlgorithm()
    {
        return convAlgorithm;
    }

    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int)algo;
    }

    private:
    int convAlgorithm;
    /*inlcude 3 type：float32、double64、float16*/
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    void createHandles()
    {
        checkCUDNN(cudnnCreate(&cudnnHandle));
        checkCublasErrors(cublasCreate(&cublasHandle));
    }


    void destroyHandles()
    {
        checkCUDNN(cudnnDestroy(cudnnHandle));
        checkCublasErrors(cublasDestroy(cublasHandle));
    }



public:

    static cuDNN_netWork<T>* instanceObject()
    {
        static cuDNN_netWork<T>* cudnn = new cuDNN_netWork<T>();
        return cudnn;
    }

    cuDNN_netWork(){

        convAlgorithm = -1;

        switch(sizeof(T))
        {
            case 2:
            dataType = CUDNN_DATA_HALF;break;
            case 4:
            dataType = CUDNN_DATA_FLOAT;break;
            case 8:
            dataType = CUDNN_DATA_DOUBLE;break;

            default:FatalError("Unsupported data type");
        }

        /*format type*/
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();
    }


    ~cuDNN_netWork()
    {
        destroyHandles();
    }

};






#endif /* CUDNN_NETWORK_H_ */
