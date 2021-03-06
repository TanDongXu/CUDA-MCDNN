#include"PoolLayer.h"
#include<glog/logging.h>
#include"common/utility.cuh"

/*
 * Create CUDNN handles
 * */
void PoolLayer:: createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
}

/*
 * Destroy CUDNN Handles
 * */
void PoolLayer:: destroyHandles()
{
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc))
}

PoolLayer::~PoolLayer()
{
    MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
    MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
    destroyHandles();
    delete m_poolMethod;
}

int PoolLayer::getOutputSize()
{
    return outputSize;
}

/*
 * Pool layer constructor
 * */
PoolLayer::PoolLayer(string name)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    m_poolMethod = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    poolingDesc = NULL;

    configPooling* curConfig = (configPooling*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(prevLayerName);

    PoolingMode = (cudnnPoolingMode_t)curConfig->_poolType;
    poolDim = curConfig->_size;
    pad_h = curConfig->_pad_h;
    pad_w = curConfig->_pad_w;
    stride_h =  curConfig->_stride_h;
    stride_w = curConfig->_stride_w;

    prev_num = prev_Layer->number;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;

    inputImageDim = prev_Layer->height;
    inputAmount = prev_Layer->channels;
    number = prev_num;
    channels = prev_channels;
    height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    outputSize = channels * height * width;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")" ;
}

/*
 * Pool layer constructor overload
 */
PoolLayer::PoolLayer(string name, const param_tuple& args)
{
    std::tie(pool_Type, poolDim, pad_h, pad_w, stride_h,
             stride_w, inputImageDim, inputAmount) = args;

    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    lrate = 0.0f;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    poolingDesc = NULL;

    m_poolMethod = new ConfigPoolMethod(pool_Type);
    PoolingMode = (cudnnPoolingMode_t)m_poolMethod->getValue();
    prev_num = config::instanceObjtce()->get_batchSize();
    prev_channels = inputAmount;
    prev_height = inputImageDim;
    prev_width = inputImageDim;
    number = prev_num;
    channels = prev_channels;
    height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    outputSize = channels * height * width;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")" ;
}

/*
 * Deep copy constructor
 */
PoolLayer::PoolLayer(const PoolLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    m_poolMethod = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    poolingDesc = NULL;


    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName;
    PoolingMode = layer->PoolingMode;
    poolDim = layer->poolDim;
    pad_h = layer->pad_h;
    pad_w = layer->pad_w;
    stride_h = layer->stride_h;
    stride_w = layer->stride_w;

    prev_num = layer->prev_num;
    prev_channels = layer->prev_channels;
    prev_height = layer->prev_height;
    prev_width = layer->prev_width;

    inputImageDim = layer->inputImageDim;
    inputAmount = layer->inputAmount;
    number = layer->number;
    channels = layer->channels;
    height = layer->height;
    width = layer->width;
    outputSize = layer->outputSize;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")" ;
    cout<<"Pool-copy"<<endl;
}
/*
 * Deep copy constructor
 */
PoolLayer::PoolLayer(const configBase* templateConfig)
{
    srcData = NULL;
    dstData = NULL;
    diffData = NULL;
    m_poolMethod = NULL;
    prevLayer.clear();
    nextLayer.clear();
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    poolingDesc = NULL;

    _name = templateConfig->_name;
    _inputName = templateConfig->_input;
    configPooling* curConfig = (configPooling*) templateConfig;
    LayersBase* prev_Layer = (LayersBase*)Layers::instanceObject()->getLayer(_inputName);

    PoolingMode = (cudnnPoolingMode_t)curConfig->_poolType;
    poolDim = curConfig->_size;
    pad_h = curConfig->_pad_h;
    pad_w = curConfig->_pad_w;
    stride_h = curConfig->_stride_h;
    stride_w = curConfig->_stride_w;

    prev_num = prev_Layer->number;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;

    inputImageDim = prev_Layer->height;
    inputAmount = prev_Layer->channels;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    outputSize = channels * height * width;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, prev_num * prev_channels * prev_height * prev_width * sizeof(float));

    this->createHandles();
    LOG(INFO) << "(" << number << "," << channels << "," << height << "," << width << ")" ;
    cout<<"Pool-copy"<<endl;
}

// ReShape the demension
void PoolLayer::ReShape()
{

    configPooling* curConfig = (configPooling*) config::instanceObjtce()->getLayersByName(_name);
    LayersBase* prev_Layer = (LayersBase*)Layers::instanceObject()->getLayer(_inputName);
    prev_num = prev_Layer->number;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;

    inputImageDim = prev_Layer->height;
    inputAmount = prev_Layer->channels;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    width = static_cast<int>(ceil(static_cast<float>(inputImageDim + 2 * pad_h - poolDim)/stride_h)) + 1 ;
    outputSize = channels * height * width;

    if(curConfig->_poolType == POOLING_AVERAGE_COUNT_INCLUDE_PADDING || curConfig->_poolType == POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
    {
        if(poolDim == stride_h && poolDim == stride_w)
        {
            poolDim = curConfig->_size;
            pad_h = curConfig->_pad_h;
            pad_w = curConfig->_pad_w;
            stride_h = curConfig->_stride_h;
            stride_w = curConfig->_stride_w;
        }
    }
}

/*
 * Pool layer Forward propagation
 * */
void PoolLayer::forwardPropagation(string train_or_test)
{
    srcData = prevLayer[0]->dstData;

    // dynamic adjust demension
    ReShape();
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
                                           PoolingMode,
                                           CUDNN_PROPAGATE_NAN,
                                           poolDim,
                                           poolDim,//window
                                           pad_h,
                                           pad_w,//pading
                                           stride_h,
                                           stride_w));//stride

    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          prev_num,
                                          prev_channels,
                                          prev_height,
                                          prev_width));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          number,
                                          channels,
                                          height,
                                          width));

    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnPoolingForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                   poolingDesc,
                                   &alpha,
                                   srcTensorDesc,
                                   srcData,
                                   &beta,
                                   dstTensorDesc,
                                   dstData));

}

/*
 * Pool layer Backward propagation
 * */
void PoolLayer::backwardPropagation(float Momentum)
{
    float alpha = 1.0f;
    float beta = 0.0;
    int nIndex = m_nCurBranchIndex;
    checkCUDNN(cudnnPoolingBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                    poolingDesc,
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

