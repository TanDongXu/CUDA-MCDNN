#include"LRNLayer.h"
#include"../common/checkError.h"
#include"../cuDNN_netWork.h"
#include<glog/logging.h>

/*
 * Create CUDNN Handles
 * */
void LRNLayer::createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateLRNDescriptor(&normDesc));
}

/*
 * Destroy CUDNN Handles
 * */
void LRNLayer::destroyHandles()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
    checkCUDNN(cudnnDestroyLRNDescriptor(normDesc));
}

/*
 * Get outputSize
 * */
int LRNLayer::getOutputSize()
{
    return outputSize;
}

/*
 * LRN layer constructor
 * */
LRNLayer::LRNLayer(string name)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    configLRN* curConfig = (configLRN*)config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*)Layers::instanceObject()->getLayer(prevLayerName);

    lrnN = curConfig->_lrnN;
    lrnAlpha = curConfig->_lrnAlpha;
    lrnBeta = curConfig->_lrnBeta;
    lrnK = 1.0;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    inputSize = prev_Layer->getOutputSize();
    outputSize =inputSize;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, number * channels * height* width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number  << "," << channels << "," << height << "," << width << ")";
}

/*
 * Deep copy constructor
 */
LRNLayer::LRNLayer(const LRNLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName;

    lrnN = layer->lrnN;
    lrnAlpha = layer->lrnAlpha;
    lrnBeta = layer->lrnBeta;
    lrnK = layer->lrnK;

    inputAmount = layer->inputAmount;
    inputImageDim = layer->inputImageDim;
    number = layer->number;
    channels = layer->channels;
    height = layer->height;
    width = layer->width;
    inputSize = layer->inputSize;
    outputSize = layer->outputSize;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number  << "," << channels << "," << height << "," << width << ")";
}

/*
 * Deep copy constructor
 */
LRNLayer::LRNLayer(const configBase* templateConfig)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;

    _name = templateConfig->_name;
    _inputName = templateConfig->_input;
    configLRN* curConfig = (configLRN*) templateConfig;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(_inputName);
    lrnN = curConfig->_lrnN;
    lrnAlpha = curConfig->_lrnAlpha;
    lrnBeta = curConfig->_lrnBeta;
    lrnK = 1.0;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    inputSize = prev_Layer->getOutputSize();
    outputSize = inputSize;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number  << "," << channels << "," << height << "," << width << ")";
}

/*
 * Destructor
 * */
LRNLayer::~LRNLayer()
{
    MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
    MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
    destroyHandles();
}

// ReShape the demesion
void LRNLayer::ReShape()
{
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(_inputName);
    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    inputSize = prev_Layer->getOutputSize();
    outputSize = inputSize;
}

/*
 * LRN Forward propagation
 * */
void LRNLayer::forwardPropagation(string train_or_test)
{
    srcData = prevLayer[0]->dstData;

    // dynamic adjust demension
    ReShape();
    checkCUDNN(cudnnSetLRNDescriptor(normDesc,
                                     lrnN,
                                     lrnAlpha,
                                     lrnBeta,
                                     lrnK));

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

    float alpha = 1.0f;
    float beta = 0.0f;
    checkCUDNN(cudnnLRNCrossChannelForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                           normDesc,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alpha,
                                           srcTensorDesc,
                                           srcData,
                                           &beta,
                                           dstTensorDesc,
                                           dstData));
}

/*
 * LRN Backward propagation
 * */
void LRNLayer::backwardPropagation(float Momentum)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    int nIndex = m_nCurBranchIndex;
    checkCUDNN(cudnnLRNCrossChannelBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
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

