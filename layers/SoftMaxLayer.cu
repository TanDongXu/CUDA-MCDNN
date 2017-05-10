#include"SoftMaxLayer.h"
#include<math.h>

/*
 * Create CUDNN Handles
 * */
void SoftMaxLayer::createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
}

/*
 * Destroy the CUDNN Handles
 * */
void SoftMaxLayer:: destroyHandles()
{
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
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
    m_devGroundTruth = NULL;
    m_hostGroundTruth = NULL;
    dev_logArray = NULL;
    nextLayer.clear();
    prevLayer.clear();
    flag = 0;
    dataSize = 0;
    CorrectSize = 0;
    cur_correctSize = 0;

    configSoftMax* curConfig = (configSoftMax*) config::instanceObjtce()->getLayersByName(_name);
    _inputName = curConfig->_input;
    LayersBase* prev_Layer =(LayersBase*) Layers::instanceObject()->getLayer(_inputName);

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

    host_result = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height * width * sizeof(float));
    m_hostGroundTruth = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height *width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &srcDiff, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &m_devGroundTruth, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_logArray, number * channels * height * width * sizeof(float));

    this->createHandles();
}

/*
 * Deep copy constructor
 * */
SoftMaxLayer::SoftMaxLayer(const SoftMaxLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    srcDiff = NULL;
    diffData = NULL;
    devLabel = NULL;
    srcDiff = NULL;
    srcLabel = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    dev_logArray = NULL;
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
    m_hostGroundTruth = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(number * channels * height *width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &srcDiff, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &devLabel, batchSize * 1 * 1 * 1 * sizeof(int));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &m_devGroundTruth, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_logArray, number * channels * height * width * sizeof(float));
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

// ReShape the demension
void SoftMaxLayer::ReShape()
{
    LayersBase* prev_Layer =(LayersBase*) Layers::instanceObject()->getLayer(_inputName);
    inputSize = prev_Layer->getOutputSize();
    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
}

// compute array log
__global__ void compute_array_log(float* array, float* groundTruth, int size)
{
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for(int i = 0; i < size; i += num_threads)
    {
        int index = i + thread_index;
        if(index < size)
        {
            groundTruth[index] = log(array[index]) * groundTruth[index];
        }
    }
}


void SoftMaxLayer::compute_cost()
{
    MemoryMonitor::instanceObject()->cpuMemoryMemset(m_hostGroundTruth, number * channels * height * width * sizeof(float));
    
    const int max_digit = nclasses;
    for(int i = 0; i < batchSize; i++)
    {
        int index = srcLabel[i];
        m_hostGroundTruth[i * max_digit + index] = 1.0f;
    }

    MemoryMonitor::instanceObject()->cpu2Gpu(m_devGroundTruth, m_hostGroundTruth, number * channels * height * width * sizeof(float));
    int size = number * channels * height * width;
    int threadsPerBlock = 256;
    int blockPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    compute_array_log<<<blockPerGrid, threadsPerBlock>>>(dstData, m_devGroundTruth, size);
    cudaThreadSynchronize();
    MemoryMonitor::instanceObject()->gpu2cpu(m_hostGroundTruth, m_devGroundTruth, number * channels * height * width * sizeof(float));
    float tmpSum = 0.0f;
    for(int i = 0; i < batchSize * channels; i++)
    {
        tmpSum += m_hostGroundTruth[i];
    }
    m_fCost = -tmpSum / batchSize;
}

/*
 * Softmax layer forward propagation
 * */
void SoftMaxLayer::forwardPropagation(string train_or_test)
{
    GetDataSize_BatchLabel();
    srcData = prevLayer[0]->dstData;

    // dynamic adjust demension
    ReShape();
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
    else
        compute_cost();
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
                                    dstTensorDesc,
                                    srcDiff,
                                    &beta,
                                    srcTensorDesc,
                                    diffData));
}

