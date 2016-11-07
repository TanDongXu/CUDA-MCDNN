#include"SoftMaxLayer.h"

/*
 * Create CUDNN Handles
 * */
void SoftMaxLayer::createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&srcDiffTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstDiffTensorDesc));
}

/*
 * Destroy the CUDNN Handles
 * */
void SoftMaxLayer:: destroyHandles()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcDiffTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstDiffTensorDesc));
}

/*
 * Destructor
 * */
SoftMaxLayer::~SoftMaxLayer()
{
    MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
    MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
    MemoryMonitor::instanceObject()->freeGpuMemory(srcDiff);
    MemoryMonitor::instanceObject()->freeGpuMemory(devLabel);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_result);
    destroyHandles();
}

/*
 * Get outputSize
 * */
int SoftMaxLayer::getOutputSize()
{
	return 0;
}

/*
 * Get the correct number
 * */
int SoftMaxLayer::getCorrectNum()
{
    return CorrectSize;
}

/*
 * Get the datasize and label
 * */
void SoftMaxLayer::GetDataSize_BatchLabel()
{
    DataLayer* data_Layer = (DataLayer*) Layers::instanceObject()->getLayer("data");
    dataSize = data_Layer->getDataSize();
    srcLabel = data_Layer->getDataLabel();
}

/*
 * SoftMax layer constructor
 * */
SoftMaxLayer::SoftMaxLayer(string name)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    srcDiff = NULL;
    diffData = NULL;
    devLabel = NULL;
    srcDiff = NULL;
    host_result = NULL;
    srcLabel = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    srcDiffTensorDesc = NULL;
    dstDiffTensorDesc = NULL;
    nextLayer.clear();
    prevLayer.clear();
    flag = 0;
    dataSize = 0;
    CorrectSize = 0;
    cur_correctSize = 0;

    configSoftMax* curConfig = (configSoftMax*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer =(LayersBase*) Layers::instanceObject()->getLayer(prevLayerName);

    batchSize = config::instanceObjtce()->get_batchSize();
    inputSize = prev_Layer->getOutputSize();
    nclasses = curConfig->_nclasses;
    lambda = curConfig->_weight_decay;
    outputSize = nclasses;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;

    host_result = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width *sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &srcDiff, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));

    this->createHandles();
}

/*
 * Deep copy constructor
 * */
SoftMaxLayer::SoftMaxLayer(SoftMaxLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    srcDiff = NULL;
    diffData = NULL;
    devLabel = NULL;
    srcDiff = NULL;
    host_result = NULL;
    srcLabel = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    srcDiffTensorDesc = NULL;
    dstDiffTensorDesc = NULL;
    nextLayer.clear();
    prevLayer.clear();
    flag = 0;
    dataSize = 0;
    CorrectSize = 0;
    cur_correctSize = 0;

    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName;
    batchSize = layer->batchSize;
    inputSize = layer->inputSize;
    nclasses = layer->nclasses;
    lambda = layer->lambda;
    outputSize = layer->outputSize;

    inputAmount = layer->inputAmount;
    inputImageDim = layer->inputImageDim;
    number = layer->number;
    channels = layer->channels;
    height = layer->height;
    width = layer->width;

    host_result = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &srcDiff, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));
    this->createHandles();
}

/*
 * Classification results
 * */
void SoftMaxLayer::ClassificationResults()
{
    if(flag == 0)
    {
        cur_correctSize = dataSize;
    }

    const int max_digit = nclasses;
    MemoryMonitor::instanceObject()->gpu2cpu(host_result, dstData, number * channels * height * width * sizeof(float));

    int temp = ((number <= dataSize - flag) ? number : (dataSize-flag));
    for(int i = 0; i < temp; i++)
    {
        float max = host_result[i * max_digit];
        int labelIndex = 0;
        for(int j = 1; j < max_digit;j++)
        {
            if(max < host_result[i * max_digit + j])
            {
                max = host_result[i * max_digit + j];
                labelIndex = j;
            }
        }
        flag++;
        if(srcLabel[i] != labelIndex) --cur_correctSize;
    }

    if(flag == dataSize)
    {
        cout<< _name << " " << cur_correctSize << "/" << CorrectSize <<" ";
        if(cur_correctSize > CorrectSize)
        {
            CorrectSize = cur_correctSize;
            //saveNetWork();
        }
        flag = 0;
    }
}

/*
 * Softmax layer forward propagation
 * */
void SoftMaxLayer::forwardPropagation(string train_or_test)
{
    GetDataSize_BatchLabel();
    srcData = prevLayer[0]->dstData;

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
    checkCUDNN(cudnnSoftmaxForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_CHANNEL,
                                   &alpha,
                                   srcTensorDesc,
                                   srcData,
                                   &beta,
                                   dstTensorDesc,
                                   dstData));

    if(train_or_test == "test" )
        ClassificationResults();
}

/*
 * Compute the diff
 * */
__global__ void SoftmaxLossBackprop(const int* label, int num_labels, int batch_size, float* diffData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    const int label_value = label[idx];
    /* For each item in the batch, decrease the result of the label's value by 1*/
    diffData[idx * num_labels + label_value] -= 1.0f;
}


/*
 * Compute diff
 * */
void SoftMaxLayer::getBackPropDiffData()
{
    MemoryMonitor::instanceObject()->cpu2Gpu(devLabel, srcLabel, batchSize * 1 * 1 * 1 * sizeof(int));
    MemoryMonitor::instanceObject()->gpu2gpu(srcDiff, dstData, number * channels * height * width * sizeof(float));

    SoftmaxLossBackprop<<< (batchSize + 127)/128, 128>>>(devLabel, nclasses, batchSize, srcDiff);
    cudaThreadSynchronize();
}

/*
 * SoftMAX backward propagation
 * */
void SoftMaxLayer::backwardPropagation(float Momentum)
{
    getBackPropDiffData();
    checkCUDNN(cudnnSetTensor4dDescriptor(srcDiffTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          number,
                                          channels,
                                          height,
                                          width));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstDiffTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          number,
                                          channels,
                                          height,
                                          width));

    float alpha = 1.0f;
    float beta = 0.0f;
    /*
     * Computes the gridient of the softmax
     * */
    checkCUDNN(cudnnSoftmaxBackward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    dstTensorDesc,
                                    dstData,
                                    srcDiffTensorDesc,
                                    srcDiff,
                                    &beta,
                                    dstDiffTensorDesc,
                                    diffData));
}

