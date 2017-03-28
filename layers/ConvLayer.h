/*
* ConvLayer.h
*
*  Created on: Nov 23, 2015
*      Author: tdx
*/

#ifndef CONVLAYER_H_
#define CONVLAYER_H_

#include"./LayersBase.h"
#include"../common/cuMatrixVector.h"
#include"../common/cuMatrix.h"
#include"../common/utility.cuh"
#include "../cuDNN_netWork.h"
#include"../config/config.h"
#include"../common/MemoryMonitor.h"
#include"../common/checkError.h"
#include<time.h>
#include<tuple>
#include<cudnn.h>
#include<curand.h>

using namespace std;

/*
 * Class convolution Layer
 * */
class ConvLayer:public LayersBase
{
    public:
    typedef tuple<int, int, int, int, int, int, int, int, float, float, float> param_tuple;

    ConvLayer(string name, int sign);
    ConvLayer(string name, int sign, const param_tuple& args);
    ConvLayer(const ConvLayer* layer);
    ConvLayer(const configBase* templateConfig);
    ~ConvLayer();
    void initRandom();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file);
    void readWeight(FILE*file);
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data );
    void createHandles();
    void destroyHandles();
    int getOutputSize();

    private:
    float *host_Weight, *dev_Weight;
    float *tmp_Wgrad, *tmp_Bgrad;
    float *host_Bias, *dev_Bias;
    float *dev_Wgrad, *dev_Bgrad;
    float lambda;
    float epsilon;
    int kernelSize;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int kernelAmount;
    int outputSize;
    int batchSize;
    int prev_num;
    int prev_channels;
    int prev_height;
    int prev_width;

private:
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnTensorDescriptor_t biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convFwdAlgo;
    cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
    cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
    curandGenerator_t curandGenerator_W;
    curandGenerator_t curandGenerator_B;
};



#endif /* CONVLAYER_H_ */
