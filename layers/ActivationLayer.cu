#include"ActivationLayer.h"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include"../common/utility.cuh"
#include<glog/logging.h>

/*
 * Create CUDNN handles
 */
void ActivationLayer::createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
}

/*
 * Destroy CUDNN Handles
 * */
void ActivationLayer::destroyHandles()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
}

/*
 * Get the outpues size
 * */
int ActivationLayer::getOutputSize()
{
   return outputSize;
}

/*
 * Activation layer constructor
 * */
ActivationLayer::ActivationLayer(string name)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    activDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    configActivation * curConfig = (configActivation*) config::instanceObjtce()->getLayersByName(_name);
    string preLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(preLayerName);

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    outputSize = channels * height * width;
    ActivationMode = curConfig->_non_linearity;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")";
}

/*
 * Deep copy constructor
 */
ActivationLayer::ActivationLayer(const ActivationLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    activDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName;
    inputAmount = layer->inputAmount;
    inputImageDim = layer->inputImageDim;
    number = layer->number;
    channels =  layer->channels;
    height = layer->height;
    width = layer->width;
    outputSize = layer->outputSize;
    ActivationMode = layer->ActivationMode;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")";
    cout<<"Activation-copy"<<endl;
}

/*
 * Deep copy constructor
 */
ActivationLayer::ActivationLayer(const configBase* templateConfig)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    activDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    _name = templateConfig->_name;
    _inputName = templateConfig->_input;
    configActivation* curConfig = (configActivation*) templateConfig;
    LayersBase* prev_Layer = (LayersBase*)Layers::instanceObject()->getLayer(_inputName);
    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels =  prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    outputSize = channels * height * width;
    ActivationMode = curConfig->_non_linearity;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")";
    cout<<"Activation-copy"<<endl;
}

/*
 * Destructor
 */
ActivationLayer::~ActivationLayer()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
	destroyHandles();
}

/*
 * LRELU activation function forward compute
*/
__global__ void LreluForward(float* srcData, float* dstData, int data_size)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for(int i = 0; i < data_size; i += num_threads)
    {
        int index = i + thread_index;
        if(index < data_size)
        {
            dstData[index] = srcData[index] > 0 ? srcData[index] : srcData[index] * 0.01;
        }
    }

}

// ReShape the demension
void ActivationLayer::ReShape()
{
    
    LayersBase* prev_Layer = (LayersBase*)Layers::instanceObject()->getLayer(_inputName);
    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels =  prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    outputSize = channels * height * width;
}

/*
 * Activation forward propagation
 * */
void ActivationLayer::forwardPropagation(string train_or_test)
{
    srcData = prevLayer[0]->dstData;

    // dynamic adjust demension
    ReShape();
    if(ActivationMode == ACTIVATION_LRELU)
    {
        int data_size = number * channels * height * width;
        int num_threads = 256;
        int num_block = (data_size + num_threads - 1) / num_threads;

        LreluForward<<<num_block, num_threads>>>(srcData, dstData, data_size);
        cudaThreadSynchronize();
    }
    else
    {
        cudnnActivationMode = (cudnnActivationMode_t)ActivationMode;
        checkCUDNN(cudnnSetActivationDescriptor(activDesc,
        		                                cudnnActivationMode,
        		                                CUDNN_PROPAGATE_NAN,
        		                                0.0));

        checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                              number,
                                              channels,
                                              height,
                                              width));

        checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                              cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                              cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                              number,
                                              channels,
                                              height,
                                              width));

        MemoryMonitor::instanceObject()->gpuMemoryMemset(dstData,number*channels*height*width*sizeof(float));
        float alpha = 1.0f;
        float beta = 0.0f;
        checkCUDNN(cudnnActivationForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                          activDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          dstData));
    }
}

/*
 * LRELU BackWard Compute
*/
__global__ void LreluBackward(float* srcDiff, float* dstDiff, float* srcData, int data_size)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = 0; i < data_size; i += num_threads)
    {
        int index = i + thread_index;
        if(index < data_size)
        {
            dstDiff[index] = srcDiff[index] * ((srcData[index] > 0) + (srcData[index] <= 0) * 0.01);
        }
    }

}

/*
 * Activation Backward Propagation
 * */
void ActivationLayer::backwardPropagation(float Momentum)
{
    if(ActivationMode == ACTIVATION_LRELU)
    {
        int nIndex = m_nCurBranchIndex;
        int data_size = number * channels * height * width;
        int num_threads = 256;
        int num_block = (data_size + num_threads - 1) / num_threads;

        LreluBackward<<<num_block, num_threads>>>(nextLayer[nIndex]->diffData, diffData, srcData, data_size);
        cudaThreadSynchronize();
    }
    else
    {
        cudnnActivationMode = (cudnnActivationMode_t)ActivationMode;
        float alpha = 1.0f;
        float beta = 0.0f;
        int nIndex = m_nCurBranchIndex;
        checkCUDNN(cudnnActivationBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                           activDesc,
                                           &alpha,
                                           dstTensorDesc,
                                           dstData,
                                           dstTensorDesc,
                                           nextLayer[nIndex]->diffData,
                                           srcTensorDesc,
                                           srcData,
                                           &beta,
                                           srcTensorDesc,
                                           diffData));
    }
}

