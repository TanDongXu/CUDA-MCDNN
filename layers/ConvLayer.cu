#include"ConvLayer.h"
#include<cuda_runtime_api.h>
#include<glog/logging.h>

/*
 * Create handles
 * */
void ConvLayer::createHandles()
{
    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    curandCreateGenerator(&curandGenerator_W, CURAND_RNG_PSEUDO_MTGP32);
    curandCreateGenerator(&curandGenerator_B, CURAND_RNG_PSEUDO_MTGP32);
}

/*
 * Destroy the handles
 * */
void ConvLayer:: destroyHandles()
{
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    curandDestroyGenerator(curandGenerator_W);
   	curandDestroyGenerator(curandGenerator_B);
}

/*
 * Get the outputSize
 * */
int ConvLayer::getOutputSize()
{
   return outputSize;
}

/*
 * Random initial weights and Bias
 * */
void ConvLayer::initRandom()
{
    srand((unsigned)time(NULL));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));

    //set seed
    curandSetPseudoRandomGeneratorSeed(curandGenerator_W, time(NULL));
    curandSetPseudoRandomGeneratorSeed(curandGenerator_B, time(NULL));
    curandGenerateNormal(curandGenerator_W, dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize, 0, epsilon);
    //curandGenerateNormal(curandGenerator_B, dev_Bias, kernelAmount, 0, 0);

   // float* tmpWeight;
   // tmpWeight = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));

   // for(int n = 0; n < kernelAmount; n++)
   // {
   //     for(int c = 0; c < inputAmount; c++)
   //     {
   //         for(int h = 0; h < kernelSize; h++)
   //         {
   //             for(int w = 0; w < kernelSize; w++)
   //             {
   //                 tmpWeight[w + kernelSize * h + kernelSize * kernelSize * c + kernelSize * kernelSize * inputAmount * n] = epsilon * (2.0f * rand() / RAND_MAX - 1.0f);
   //             }
   //         }
   //     }
   // }

   // MemoryMonitor::instanceObject()->cpu2Gpu(dev_Weight, tmpWeight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));

   // delete tmpWeight;
}

/*
 * ConvLayer constructor
 * */
ConvLayer::ConvLayer(string name, int sign)
{
    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    host_Weight = NULL;
    host_Bias = NULL;
    dev_Weight = NULL;
    dev_Bias = NULL;
    dev_Wgrad = NULL;
    dev_Bgrad = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;
    diffData = NULL;
    dev_weightSquare = NULL;
    host_weightSquare = NULL;
    prevLayer.clear();
    nextLayer.clear();

    filterDesc = NULL;
    convDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    biasTensorDesc = NULL;
    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;

    /*can use class member prevLayer, because it has not assignment*/
    configConv* curConfig = (configConv*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(prevLayerName);

    epsilon = curConfig->_init_w;
    lrate = curConfig->_lrate;
    batchSize = config::instanceObjtce()->get_batchSize();
    kernelAmount = curConfig->_kernelAmount;
    kernelSize = curConfig->_kernelSize;
    pad_h = curConfig->_pad_h;
    pad_w = curConfig->_pad_w;
    stride_h = curConfig->_stride_h;
    stride_w = curConfig->_stride_w;
    lambda = curConfig->_weight_decay;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    prev_num = prev_Layer->number;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;
    number = prev_num;
    channels = kernelAmount;
    height = (inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;
    width = (inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;
    outputSize = channels * height * width;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, batchSize * kernelAmount * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, batchSize * inputAmount * inputImageDim * inputImageDim * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    host_weightSquare = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));

    this->createHandles();
    if(sign == RANDOM)
        this->initRandom();

    LOG(INFO) << "(" << number <<","<< channels << "," << height << "," << width <<")";
}

/*
 * Conv constructor overload
 * */
ConvLayer::ConvLayer(string name, int sign, const param_tuple& args)
{
    std::tie(pad_h, pad_w, stride_h, stride_w, kernelSize,
             kernelAmount, inputAmount, inputImageDim,
             epsilon, lrate, lambda) = args;

    _name = name;
    _inputName = " ";
    srcData = NULL;
    dstData = NULL;
    host_Weight = NULL;
    host_Bias = NULL;
    dev_Weight = NULL;
    dev_Bias = NULL;
    dev_Wgrad = NULL;
    dev_Bgrad = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;
    diffData = NULL;
    dev_weightSquare = NULL;
    host_weightSquare = NULL;
    prevLayer.clear();
    nextLayer.clear();

    filterDesc = NULL;
    convDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    biasTensorDesc = NULL;
    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;

    batchSize = config::instanceObjtce()->get_batchSize();
    prev_num = config::instanceObjtce()->get_batchSize();
    prev_channels = inputAmount;
    prev_height = inputImageDim;
    prev_width = inputImageDim;
    number = prev_num;
    channels = kernelAmount;
    height = (inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;
    width = (inputImageDim + 2 * pad_h - kernelSize)/stride_h + 1;
    outputSize = channels * height * width;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &tmp_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, batchSize * kernelAmount * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &diffData, batchSize * inputAmount * inputImageDim * inputImageDim * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    host_weightSquare = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));

    this->createHandles();
    if(sign == RANDOM)
        this->initRandom();
    LOG(INFO) << "(" << number <<","<< channels << "," << height << "," << width <<")";
}

/*
 * Deep copy constructor for convolution layers
 */
ConvLayer::ConvLayer(const ConvLayer* layer)
{
    srcData = NULL;
    dstData = NULL;
    host_Weight = NULL;
    host_Bias = NULL;
    dev_Weight = NULL;
    dev_Bias = NULL;
    dev_Wgrad = NULL;
    dev_Bgrad = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;
    diffData = NULL;
    dev_weightSquare = NULL;
    host_weightSquare = NULL;
    prevLayer.clear();
    nextLayer.clear();

    filterDesc = NULL;
    convDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    biasTensorDesc = NULL;
    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;

    static int idx = 0;
    _name = layer->_name + string("_") + int_to_string(idx);
    idx ++;
    _inputName = layer->_inputName ;
    epsilon = layer->epsilon;
    lrate = layer->lrate;
    batchSize = layer->batchSize;
    kernelAmount = layer->kernelAmount;
    kernelSize = layer->kernelSize;
    pad_h = layer->pad_h;
    pad_w = layer->pad_w;
    stride_h = layer->stride_h;
    stride_w = layer->stride_w;
    lambda = layer->lambda;
    inputAmount = layer->inputAmount;
    inputImageDim = layer->inputImageDim;
    prev_num = layer->prev_num;
    prev_channels = layer->prev_channels;
    prev_height = layer->prev_height;
    prev_width = layer->prev_width;
    number = layer->number;
    channels = layer->channels;
    height = layer->height;
    width = layer->width;
    outputSize = layer->outputSize;

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, batchSize * kernelAmount * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, batchSize * inputAmount * inputImageDim * inputImageDim * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    host_weightSquare = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    //    MemoryMonitor::instanceObject()->gpu2gpu(dev_Wgrad, layer->dev_Wgrad, kernelAmount * inputAmount * 1 * kernelSize * kernelSize * sizeof(float));
    //    MemoryMonitor::instanceObject()->gpu2gpu(dev_Bgrad, layer->dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    
    epsilon = 0.1;// use for cifar100
    this->createHandles();
    this->initRandom();

    //use for cifar100
    float* tWeight = NULL;
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tWeight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpu2gpu(tWeight, layer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    int size = kernelAmount * inputAmount * kernelSize * kernelSize;
    float alpha = 1.0f;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                 size,
                                 &alpha,
                                 tWeight,
                                 1,
                                 dev_Weight,
                                 1));
    
    LOG(INFO) << "(" << number <<","<< channels << "," << height << "," << width <<")";
    cout<<"Conv-copy"<<endl;
}


/*
 * Deep copy constructor for convolution layers
 */
ConvLayer::ConvLayer(const configBase* templateConfig)
{
    srcData = NULL;
    dstData = NULL;
    host_Weight = NULL;
    host_Bias = NULL;
    dev_Weight = NULL;
    dev_Bias = NULL;
    dev_Wgrad = NULL;
    dev_Bgrad = NULL;
    tmp_Wgrad = NULL;
    tmp_Bgrad = NULL;
    diffData = NULL;
    dev_weightSquare = NULL;
    host_weightSquare = NULL;
    prevLayer.clear();
    nextLayer.clear();

    filterDesc = NULL;
    convDesc = NULL;
    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    biasTensorDesc = NULL;
    convFwdAlgo = (cudnnConvolutionFwdAlgo_t)-1;
    convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)-1;
    convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)-1;

    _name = templateConfig->_name;
    _inputName = templateConfig->_input;
    configConv* curConfig = (configConv*) templateConfig;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(_inputName);

    epsilon = curConfig->_init_w;
    lrate = curConfig->_lrate;
    batchSize = config::instanceObjtce()->get_batchSize();
    kernelAmount = curConfig->_kernelAmount;
    kernelSize = curConfig->_kernelSize;
    pad_h = curConfig->_pad_h;
    pad_w = curConfig->_pad_w;
    stride_h = curConfig->_stride_h;
    stride_w = curConfig->_stride_w;
    lambda = curConfig->_weight_decay;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    prev_num = prev_Layer->number;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;
    number = prev_num;
    channels = kernelAmount;
    height = (inputImageDim + 2 * pad_h - kernelSize) / stride_h + 1;
    width = (inputImageDim + 2 * pad_w - kernelSize) / stride_w + 1;
    outputSize = channels * height * width;
    
    // malloc GPU memory
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tmp_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, batchSize * kernelAmount * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, batchSize * inputAmount * inputImageDim * inputImageDim * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Wgrad, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bgrad, 1 * kernelAmount * 1 * 1 * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    host_weightSquare = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    
    //find the same demension of conv layer weight from previous LayersBase
    configBase* findConfig = const_cast<configBase*>(templateConfig);
    ConvLayer* resultLayer = NULL;
    bool bFind = false;
    while(0 != findConfig->_prev.size())
    {
       if("CONV" == findConfig->_prev[0]->_type)
        {
            resultLayer = (ConvLayer*) Layers::instanceObject()->getLayer(findConfig->_prev[0]->_name);
            //must make sure kernelAmount kernelSize inputAmount equal respectively
            if((resultLayer->inputAmount == inputAmount) && (resultLayer->kernelSize == kernelSize) && (resultLayer->kernelAmount == kernelAmount))
            {
                bFind = true;
                break;
            }
        } 

        findConfig = findConfig->_prev[0];
    }

    this->createHandles();

    if(bFind)
    {
        CHECK(resultLayer);
        if(_name == "conv5" || _name == "conv7" || _name == "conv9" || _name == "conv11")
        {
            //if(_name == "conv3") epsilon = 0.01;
            if(_name == "conv5") epsilon = 0.01;
            if(_name == "conv7") epsilon = 0.005;
            if(_name == "conv9") epsilon = 0.005;
            if(_name == "conv11") epsilon = 0.005;
            this->initRandom();
            float* tWeight = NULL;
            MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tWeight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            MemoryMonitor::instanceObject()->gpu2gpu(tWeight, resultLayer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //MemoryMonitor::instanceObject()->gpu2gpu(dev_Weight, resultLayer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //reverseArray(dev_Weight, kernelAmount, inputAmount, kernelSize, kernelSize);
           // static int index = 0;
           // float scalVal = 3;
           // if(index == 1) scalVal = 1;
           // if(index == 2) scalVal = 0.5;
           // if(index == 3) scalVal = 0.1;
           int size =  kernelAmount * inputAmount * kernelSize * kernelSize;
           // checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
           //                         size,
           //                         &scalVal,
           //                         dev_Weight,
           //                         1));
           // MemoryMonitor::instanceObject()->gpu2gpu(dev_Bias, resultLayer->dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));
            float alpha = 1.0f;
            checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                 size,
                                 &alpha,
                                 tWeight,
                                 1,
                                 dev_Weight,
                                 1));
        }else
        {
            MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));
            MemoryMonitor::instanceObject()->gpu2gpu(dev_Weight, resultLayer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //float* tWeight = NULL;
            //MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&tWeight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //MemoryMonitor::instanceObject()->gpu2gpu(tWeight, resultLayer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //MemoryMonitor::instanceObject()->gpu2gpu(dev_Weight, resultLayer->dev_Weight, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
            //reverseArray(dev_Weight, kernelAmount, inputAmount, kernelSize, kernelSize);
            //static int index = 0;
            //float scalVal = 3;
            //if(index == 1) scalVal = 1;
            //if(index == 2) scalVal = 0.5;
            //if(index == 3) scalVal = 0.1;
            //int size =  kernelAmount * inputAmount * kernelSize * kernelSize;
            //checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
            //                          size,
            //                          &scalVal,
            //                          dev_Weight,
            //                          1));
            //MemoryMonitor::instanceObject()->gpu2gpu(dev_Bias, resultLayer->dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));
            //float alpha = 0.5f;
            // checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
            //                           size,
            //                           &alpha,
            //                           tWeight,
            //                           1,
            //                           dev_Weight,
            //                           1));
        }

        MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_Bias, kernelAmount * 1 * 1 * 1 * sizeof(float));
        cout<<epsilon<<endl;
        cout<< _name<<endl;
        cout<<"copy weight"<<endl;
    }else
    {
        this->initRandom();
        cout<<"random weight"<<endl;
        //printf_DevParameter(kernelAmount,inputAmount, kernelSize, kernelSize, dev_Weight);
    }

    LOG(INFO) << "(" << number <<","<< channels << "," << height << "," << width <<")";
    cout<<"conv_copy"<<endl;
}

/*
 * Destructor
 * */
ConvLayer::~ConvLayer()
{
    MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_Weight);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bias);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_Wgrad);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_Bgrad);
    MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Wgrad);
    MemoryMonitor::instanceObject()->freeGpuMemory(tmp_Bgrad);
    MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
    MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
    destroyHandles();
}

// ReShape the demension int the Forward
void ConvLayer::ReShape()
{
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(_inputName);
    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    prev_channels = prev_Layer->channels;
    prev_height = prev_Layer->height;
    prev_width = prev_Layer->width;
    number = prev_num;
    channels = kernelAmount;
    height = (inputImageDim + 2 * pad_h - kernelSize) / stride_h + 1;
    width = (inputImageDim + 2 * pad_w - kernelSize) / stride_w + 1;
    outputSize = channels * height * width;
}

//__global__ void compute_array_square(float* array, float* outArray, int size)
//{
//    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
//    int num_threads = blockDim.x * gridDim.x;
//    for(int i = 0; i < size; i += num_threads)
//    {
//        int index = i + thread_index;
//        if(index < size)
//        {
//            outArray[index] = array[index] * array[index];
//        }
//    }
//}

//compute cost
void ConvLayer::compute_cost()
{
    MemoryMonitor::instanceObject()->gpuMemoryMemset(dev_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    MemoryMonitor::instanceObject()->cpuMemoryMemset(host_weightSquare, kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    int size = kernelAmount * inputAmount * kernelSize * kernelSize;
    int threadsPerBlock = 256;
    int blockPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    compute_array_square<<<blockPerGrid, threadsPerBlock>>>(dev_Weight, dev_weightSquare, size);
    cudaThreadSynchronize();
    MemoryMonitor::instanceObject()->gpu2cpu(host_weightSquare, dev_weightSquare, size * sizeof(float));

    float tmpSum = 0.0f;
    for(int i = 0; i < size; i++)
    {
        tmpSum += host_weightSquare[i];
    }
    m_fCost = tmpSum * lambda / 2;
    //cout<<"conv: "<<m_fCost<<endl;
}

/*
 * Forward propagation add Bias
 */
void ConvLayer::addBias(const cudnnTensorDescriptor_t& dstTensorDesc, int c, float *data )
{

    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          1,
                                          c,
                                          1,
                                          1));

    float alpha = 1.0;
    float beta = 1.0;
    checkCUDNN(cudnnAddTensor(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                              &alpha,
                              biasTensorDesc,
                              dev_Bias,
                              &beta,
                              dstTensorDesc,
                              data));
}

void ConvLayer::copyWeight()
{
    int size =  kernelAmount * inputAmount * kernelSize * kernelSize;
    if(_name == "conv6")
    {
        ConvLayer* prev_Layer = (ConvLayer*) Layers::instanceObject()->getLayer("conv5");
        
        float alpha = 1.0f;
        checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  prev_Layer->dev_Weight,
                                  1,
                                  dev_Weight,
                                  1));

        cout<<"conv6 copy"<<endl;
    }
    if(_name == "conv5")
    {
        ConvLayer* prev_Layer = (ConvLayer*) Layers::instanceObject()->getLayer("conv4");
        
        float alpha = 1.0f;
        checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  prev_Layer->dev_Weight,
                                  1,
                                  dev_Weight,
                                  1));

        cout<<"conv5 copy"<<endl;
    }
    if(_name == "conv4")
    {
        ConvLayer* prev_Layer = (ConvLayer*) Layers::instanceObject()->getLayer("conv3");
        
        float alpha = 1.0f;
        checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  prev_Layer->dev_Weight,
                                  1,
                                  dev_Weight,
                                  1));

        cout<<"conv4 copy"<<endl;
    }
    if(_name == "conv3")
    {
        ConvLayer* prev_Layer = (ConvLayer*) Layers::instanceObject()->getLayer("conv2");
        
        float alpha = 1.0f;
        checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  prev_Layer->dev_Weight,
                                  1,
                                  dev_Weight,
                                  1));

        cout<<"conv3 copy"<<endl;
    }
}

/*
 * Convolution forward propagation
 * */
void ConvLayer::forwardPropagation(string train_or_test)
{
    srcData = prevLayer[0]->dstData;
    // dynamic adjust demension
    ReShape();
    //reverseArray(dev_Weight, kernelAmount, inputAmount, kernelSize, kernelSize);
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          prev_num,
                                          prev_channels,
                                          prev_height,
                                          prev_width));

    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          kernelAmount,
                                          inputAmount,
                                          kernelSize,
                                          kernelSize));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               pad_h,
                                               pad_w,//pading
                                               stride_h,
                                               stride_w,//stride
                                               1,1,//upscale
                                               CUDNN_CROSS_CORRELATION));

    checkCUDNN(cudnnSetTensor4dDescriptor(dstTensorDesc,
                                          cuDNN_netWork<float>::instanceObject()->GetTensorFormat(),
                                          cuDNN_netWork<float>::instanceObject()->GetDataType(),
                                          number,
                                          channels,
                                          height,
                                          width));

    /*
     * Obtain the best suited algorithm for cudnnConvolutinForward
     * */
    //if (cuDNN_netWork<float>::instanceObject()->getConvFwdAlgorithm() < 0)
    //{
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       0,
                                                       &convFwdAlgo));

        //cuDNN_netWork<float>::instanceObject()->setConvolutionFwdAlgorithm(convFwdAlgo);
    //}else
    //{
    //	convFwdAlgo =(cudnnConvolutionFwdAlgo_t)cuDNN_netWork<float>::instanceObject()->getConvFwdAlgorithm();
    //}

    /*Get the amount of GPU memory for cudnnConvolutionForward*/
    size_t convFwdSizeInBytes = 0;
    void* convFwdWorkSpace = NULL;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       convFwdAlgo,
                                                       &convFwdSizeInBytes));

    if (convFwdSizeInBytes != 0)
    {
        checkCudaErrors(cudaMalloc(&convFwdWorkSpace, convFwdSizeInBytes));
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                       &alpha,
                                       srcTensorDesc,
                                       srcData,
                                       filterDesc,
                                       dev_Weight,
                                       convDesc,
                                       convFwdAlgo,
                                       convFwdWorkSpace,
                                       convFwdSizeInBytes,
                                       &beta,
                                       dstTensorDesc,
                                       dstData));

    /*add bias*/
    addBias(dstTensorDesc, channels, dstData);

    if (convFwdSizeInBytes != 0)
    {
        	checkCudaErrors(cudaFree(convFwdWorkSpace));
    }
}

/*
 * Convolution backward propagation
 * */
void ConvLayer::backwardPropagation(float Momentum)
{
    /*Get the convolutuion function gradient with respect to the bias*/
    float alpha = 1.0f;
    float beta = 0.0f;
    int nIndex = m_nCurBranchIndex;
    checkCUDNN(cudnnConvolutionBackwardBias(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                            &alpha,
                                            dstTensorDesc,
                                            nextLayer[nIndex]->diffData,
                                            &beta,
                                            biasTensorDesc,
                                            tmp_Bgrad
                                           ));

    /*Obtain the best suited algorithm for cudnnConvolutionBackwardFilter*/
    //if(cuDNN_netWork<float>::instanceObject()->getConvolutionBwdFilterAlgorithm() < 0)
    //{
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
    			                                               srcTensorDesc,
    			                                               dstTensorDesc,
    			                                               convDesc,
    			                                               filterDesc,
    			                                               CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    			                                               0,
    			                                               &convBwdFilterAlgo
    			                                               ));

    //	cuDNN_netWork<float>::instanceObject()->setConvolutionBwdFilterAlgorithm(convBwdFilterAlgo);
    //}else
    //{
    //	convBwdFilterAlgo = (cudnnConvolutionBwdFilterAlgo_t)cuDNN_netWork<float>::instanceObject()->getConvolutionBwdFilterAlgorithm();
    //}

    /*Get the GPU memory workspace for cudnnConvolutionBackwardFilter*/
    size_t convBwdFilterSizeInBytes = 0;
    void* convBwdFilterWorkSpace = NULL;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
    		                                                  srcTensorDesc,
    		                                                  dstTensorDesc,
    		                                                  convDesc,
    		                                                  filterDesc,
    		                                                  convBwdFilterAlgo,
    		                                                  &convBwdFilterSizeInBytes
    /*Alloc GPU memory*/		                                                  ));
    if(convBwdFilterSizeInBytes != 0)
    {
    	checkCudaErrors(cudaMalloc(&convBwdFilterWorkSpace, convBwdFilterSizeInBytes));
    }

   /*This function computes the convolution gradient with respect to filter coefficient using the specified algo*/
    checkCUDNN(cudnnConvolutionBackwardFilter(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              dstTensorDesc,
                                              nextLayer[nIndex]->diffData,
                                              convDesc,
                                              convBwdFilterAlgo,
                                              convBwdFilterWorkSpace,
                                              convBwdFilterSizeInBytes,
                                              &beta,
                                              filterDesc,
                                              tmp_Wgrad));

    if (convBwdFilterSizeInBytes != 0)
    {
        checkCudaErrors(cudaFree(convBwdFilterWorkSpace));
    }

    /*Obtaining the best suited algorithm for the cudnnConvolutionBackwardData*/
    //if(cuDNN_netWork<float>::instanceObject()->getConvolutionBwdDataAlgorithm() < 0)
    //{
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
    			                                            filterDesc,
    			                                            dstTensorDesc,
    			                                            convDesc,
    			                                            dstTensorDesc,
    			                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    			                                            0,
    			                                            &convBwdDataAlgo
    			                                            ));
    //	cuDNN_netWork<float>::instanceObject()->setConvolutionBwdDataAlgorithm(convBwdDataAlgo);

    //}else
    //{
    //	convBwdDataAlgo = (cudnnConvolutionBwdDataAlgo_t)cuDNN_netWork<float>::instanceObject()->getConvolutionBwdDataAlgorithm();
    //}

    /*Get the amount of GPU memory for the cudnnConvlotionBackwardData*/
    size_t convBwdDataSizeInBytes = 0;
    void* convBwdDataWorkSpace = NULL;
    /*按照接口说明srcTensorDesc应该是dstTensorDesc的,参考一个代码是用srcTensorDesc*/
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
    		                                                filterDesc,
    		                                                dstTensorDesc,
    		                                                convDesc,
    		                                                srcTensorDesc,
    		                                                convBwdDataAlgo,
    		                                                &convBwdDataSizeInBytes
    		                                                ));
    if(convBwdDataSizeInBytes != 0)
    {
    	checkCudaErrors(cudaMalloc(&convBwdDataWorkSpace, convBwdDataSizeInBytes));
    }

    //Note:if use convBwdDataAlgo above,it will return error in running.
    // convBwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    /*Compute the convolution gradient with respect to the output tensor using the specified algo*/
    alpha = 1.0f;
    beta = 0.0f;
    checkCUDNN(cudnnConvolutionBackwardData(cuDNN_netWork<float>::instanceObject()->GetcudnnHandle(),
                                            &alpha,
                                            filterDesc,
                                            dev_Weight,
                                            dstTensorDesc,
                                            nextLayer[nIndex]->diffData,
                                            convDesc,
                                            convBwdDataAlgo,
                                            convBwdDataWorkSpace,
                                            convBwdDataSizeInBytes,
                                            &beta,
                                            srcTensorDesc,
                                            diffData));

    if(convBwdDataSizeInBytes != 0)
    {
    	checkCudaErrors(cudaFree(convBwdDataWorkSpace));
    }

    //compute cost
    compute_cost();

    /*
     * Update the weights in conv layer
     *
     * */
    alpha = lambda * batchSize;
    int size =  kernelAmount * inputAmount * kernelSize * kernelSize;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  dev_Weight,
                                  1,
                                  tmp_Wgrad,
                                  1));

    float scalVal = Momentum;
    size =  kernelAmount * inputAmount * kernelSize * kernelSize;
    checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  dev_Wgrad,
                                  1));

    size = kernelAmount * 1 * 1 * 1;
    checkCublasErrors(cublasSscal(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  dev_Bgrad,
                                  1));

    scalVal = lrate * 1.0f / batchSize;
    size =  kernelAmount * inputAmount * kernelSize * kernelSize;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Wgrad,
                                  1,
                                  dev_Wgrad,
                                  1));

    scalVal = 2 * lrate * 1.0f / batchSize;
    size = kernelAmount * 1 * 1 * 1;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &scalVal,
                                  tmp_Bgrad,
                                  1,
                                  dev_Bgrad,
                                  1));

    alpha = -1.0f;
    size =  kernelAmount * inputAmount * kernelSize * kernelSize;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  dev_Wgrad,
                                  1,
                                  dev_Weight,
                                  1));

    size = kernelAmount * 1 * 1 * 1;
    checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
                                  size,
                                  &alpha,
                                  dev_Bgrad,
                                  1,
                                  dev_Bias,
                                  1));
}

/*
 * Save the weights and Bias
 * */
void ConvLayer::saveWeight(FILE*file)
{
    host_Weight = NULL; host_Bias = NULL;
    copy_DeviceToHost(dev_Weight, host_Weight, kernelAmount, inputAmount, kernelSize, kernelSize);
    copy_DeviceToHost(dev_Bias, host_Bias, 1, 1, 1, kernelAmount);

    for(int n = 0 ; n < kernelAmount; n++)
    {
        for(int c = 0; c < inputAmount; c++)
        {
            for(int h = 0; h < kernelSize; h++)
            {
                for(int w = 0; w < kernelSize; w++)
                {
                    fprintf(file, "%f ", host_Weight[w + h * kernelSize + kernelSize * kernelSize * c + kernelSize * kernelSize * inputAmount * n]);
                }
            }
        }
    }

    for(int n = 0; n < kernelAmount; n++)
    {
        fprintf(file, "%f ", host_Bias[n]);
    }

    MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
}

/*
 * Read the weights and Bias from file
 * */
void ConvLayer::readWeight(FILE*file)
{
    host_Weight = NULL; host_Bias = NULL;
    dev_Weight = NULL; dev_Bias = NULL;
    host_Weight = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * inputAmount * kernelSize * kernelSize * sizeof(float));
    host_Bias = (float*)MemoryMonitor::instanceObject()->cpuMallocMemory(kernelAmount * 1 * 1 * 1 * sizeof(float));

    for(int n = 0 ; n < kernelAmount; n++)
    {
        for(int c = 0; c < inputAmount; c++)
        {
            for(int h = 0; h < kernelSize; h++)
            {
                for(int w = 0; w < kernelSize; w++)
                {
                    fscanf(file, "%f", &host_Weight[w + h * kernelSize + kernelSize * kernelSize * c + kernelSize * kernelSize * inputAmount * n]);
                }
            }
        }
    }

    for (int n = 0; n < kernelAmount; n++)
    {
        fscanf(file, "%f", &host_Bias[n]);
    }

    copy_HostToDevice(host_Weight, dev_Weight, kernelAmount, inputAmount, kernelSize, kernelSize);
    copy_HostToDevice(host_Bias, dev_Bias, 1, 1, 1, kernelAmount);

    MemoryMonitor::instanceObject()->freeCpuMemory(host_Weight);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_Bias);
}
