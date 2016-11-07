#include"DropOutLayer.h"

/*
 * DropOut layer constructor
 * */
DropOutLayer::DropOutLayer(string name)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    nextLayer.clear();
    prevLayer.clear();
    outputPtr = NULL;

    configDropOut* curConfig = (configDropOut*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(prevLayerName);

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = prev_Layer->channels;
    height = prev_Layer->height;
    width = prev_Layer->width;
    outputSize = channels * height * width;
    DropOut_rate = curConfig->dropOut_rate;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &outputPtr, number * channels * height * width * sizeof(float));
    this->createHandles();
}

/*
 * overload constructor
 * */
DropOutLayer::DropOutLayer(DropOutLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    nextLayer.clear();
    prevLayer.clear();
    outputPtr = NULL;

    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName;

    inputAmount = layer->inputAmount;
    inputImageDim = layer->inputImageDim;
    number = layer->number;
    channels = layer->channels;
    height = layer->height;
    width = layer->width;
    outputSize = layer->outputSize;
    DropOut_rate = layer->DropOut_rate;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &outputPtr, number * channels * height * width * sizeof(float));

    this->createHandles();

}

/*
 * Destructor
 * */
DropOutLayer::~DropOutLayer()
{
    MemoryMonitor::instanceObject()->freeGpuMemory(outputPtr);
    destroyHandles();
}

/*
 * Get the outputSize
 * */
int DropOutLayer::getOutputSize()
{
    return outputSize;
}

/*
 * Create Random handle
 * */
void DropOutLayer::createHandles()
{
    curandCreateGenerator(&curandGenerator_DropOut, CURAND_RNG_PSEUDO_MTGP32);
}

__global__ void dropout_train(float* data, float* outputPtr, int size, float probability)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
    {
        if(outputPtr[idx] < probability)
        data[idx] = 0;
    }
}

__global__ void dropout_test(float* data, int size, float probability)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size)
    {
        data[idx] = data[idx] * probability;
    }
}

void DropOutLayer::CreateUniform(int size)
{
    curandSetPseudoRandomGeneratorSeed(curandGenerator_DropOut, time(NULL));
    curandGenerateUniform(curandGenerator_DropOut, outputPtr, size);
}

void DropOutLayer::Dropout_TrainSet(float* data, int size, float dropout_rate)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dropout_train<<<blocksPerGrid, threadsPerBlock>>>(data, outputPtr, size, dropout_rate);
    cudaThreadSynchronize();
}

void DropOutLayer::Dropout_TestSet(float* data, int size, float dropout_rate)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    dropout_test<<<blocksPerGrid, threadsPerBlock>>>(srcData, number * channels * height * width, DropOut_rate);
    cudaThreadSynchronize();
}

void DropOutLayer::forwardPropagation(string train_or_test)
{
    srcData =prevLayer[0]->dstData;
    dstData = srcData;

    /*use dropout in training, when testing multiply probability*/
    if(train_or_test == "train")
    {
        CreateUniform(number * channels * height * width);
        Dropout_TrainSet(srcData, number * channels * height * width, DropOut_rate);
    }
    else
    Dropout_TestSet(srcData, number * channels * height * width, DropOut_rate);
}


void DropOutLayer::backwardPropagation(float Momemtum)
{
    int nIndex = m_nCurBranchIndex;
    diffData = nextLayer[nIndex]->diffData;
    Dropout_TrainSet(diffData, number * channels * height * width, DropOut_rate);
}

void DropOutLayer::destroyHandles()
{
    curandDestroyGenerator(curandGenerator_DropOut);

}
