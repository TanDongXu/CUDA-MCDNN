#include"Concat.h"


void Concat::concatInit()
{
    host_offset = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
    separateDim = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
    host_channels = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));

    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dstData, number * channels * height * width * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_offset, 4 * sizeof(int));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &dev_channels, 4 * sizeof(int));
}

/*
 * Destructor
 * */
Concat::~Concat()
{
    MemoryMonitor::instanceObject()->freeCpuMemory(host_offset);
    MemoryMonitor::instanceObject()->freeCpuMemory(host_channels);
    MemoryMonitor::instanceObject()->freeCpuMemory(separateDim);
    MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
    MemoryMonitor::instanceObject()->freeGpuMemory(diffData);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_offset);
    MemoryMonitor::instanceObject()->freeGpuMemory(dev_channels);
    MemoryMonitor::instanceObject()->freeGpuMemory(separate_diffData);
    prevDiff.vector_clear();
    separate_dstData.vector_clear();
}

/*incetion concat*/
Concat::Concat(Layers*& Inner_Layers, const param_tuple& args)
{
    std::tie(one, three, five, pool_proj) = args;
    size = 0;
    host_offset = NULL;
    dev_offset = NULL;
    separateDim = NULL;
    host_channels = NULL;
    dev_channels = NULL;
    separate_diffData = NULL;
    diffData = NULL;
    dstData = NULL;
    InnerLayers = Inner_Layers;

    /*prevLayer dim*/
    prev_num = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer[0]->number;
    prev_channels = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer[0]->channels;
    prev_height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer[0]->height;
    prev_width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer[0]->width;
    size  = prev_num * prev_channels * prev_height * prev_width;

    number = prev_num;
    channels = one + three + five + pool_proj;
    height = prev_height;
    width = prev_width;

    this->concatInit();

    host_offset[0] = 0;
    host_offset[1] = one * height * width;
    host_offset[2] = (one + three) * height * width;
    host_offset[3] = (one + three + five) * height * width;

    separateDim[0] = number * one * height * width;
    separateDim[1] = number * three * height * width;
    separateDim[2] = number * five * height * width;
    separateDim[3] = number * pool_proj * height * width;

    host_channels[0] = one;
    host_channels[1] = three;
    host_channels[2] = five;
    host_channels[3] = pool_proj;

    checkCudaErrors(cudaMemcpy(dev_offset, host_offset, 4 * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_channels, host_channels, 4 * sizeof(int), cudaMemcpyHostToDevice));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&separate_diffData, getMax<int>(separateDim, 4) * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, size * sizeof(float));

    separate_dstData.push_back(InnerLayers[0].getLayer("one")->dstData);
    separate_dstData.push_back(InnerLayers[1].getLayer("three")->dstData);
    separate_dstData.push_back(InnerLayers[2].getLayer("five")->dstData);
    separate_dstData.push_back(InnerLayers[3].getLayer("pool_proj")->dstData);
    separate_dstData.toGpu();

    prevDiff.push_back(InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->diffData);
    prevDiff.push_back(InnerLayers[1].getLayer(InnerLayers[1].getLayersName(0))->diffData);
    prevDiff.push_back(InnerLayers[2].getLayer(InnerLayers[2].getLayersName(0))->diffData);
    prevDiff.push_back(InnerLayers[3].getLayer(InnerLayers[3].getLayersName(0))->diffData);
    prevDiff.toGpu();
    }

/*inception forwardPropagation*/
float* Concat::forwardSetup()
{
    dim3 block(number, 4);
    dim3 thread(1024);
    MultiChannelsMerge<<<block,thread>>>(separate_dstData.devPoint, 
                                         dstData, 
                                         dev_channels, 
                                         dev_offset, 
                                         height, 
                                         channels);
    cudaThreadSynchronize();
    return dstData;
}


/*split the delta*/
void Concat::split_DiffData(int index, float* diffData)
{
    int curChannel = host_channels[index];
    int curOffset = host_offset[index];

    dim3 block(number);
    dim3 thread(1024);
    MultiChannelsSplit<<<block, thread>>>(diffData,
                                          separate_diffData, 
                                          curChannel, 
                                          curOffset, 
                                          height, 
                                          channels);
    cudaThreadSynchronize();
    InnerLayers[index].getLayer(InnerLayers[index].getLayersName(InnerLayers[index].getLayersNum()-1))->nextLayer[0]->diffData = separate_diffData;
}


/*inception backwardPropagation*/
float* Concat::backwardSetup()
{
    /*necessary*/
    MemoryMonitor::instanceObject()->gpuMemoryMemset(diffData, size * sizeof(float));
    dim3 block(1);
    dim3 thread(1024);
    MultiArrayAdd<<<block, thread>>>(prevDiff.devPoint, 
                                     diffData,
                                     prev_num,
                                     prev_channels, 
                                     prev_height, 
                                     prev_width);
    cudaThreadSynchronize();
    return diffData;
}
