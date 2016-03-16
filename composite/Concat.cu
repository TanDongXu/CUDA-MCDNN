#include"Concat.h"


void Concat::concatInit()
{
	host_offset = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
	separateDim = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
	host_channels = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
}

/*incetion concat*/
Concat::Concat(Layers*& Inner_Layers, const param_tuple& args)
{
	std::tie(one, three, five, pool_proj) = args;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	size = 0;
	prev_number = 0;
	prev_channels = 0;
	prev_height = 0;
	prev_width = 0;
	host_offset = NULL;
	dev_offset = NULL;
	separateDim = NULL;
	host_channels = NULL;
	dev_channels = NULL;
	separate_diffData = NULL;
	diffData = NULL;
	dstData = NULL;

	channels = one + three + five + pool_proj;
	InnerLayers = Inner_Layers;

    //init
	this->concatInit();

}

/*inception forwardPropagation*/
float* Concat::forwardSetup()
{
	number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;


	host_offset[0] = 0;
	host_offset[1] = one * height * width;
	host_offset[2] = (one + three) * height * width ;
	host_offset[3] = (one + three + five) * height * width;

	//use in next next function
	separateDim[0] = number * one * height * width;
	separateDim[1] = number * three * height * width;
	separateDim[2] = number * five * height * width;
	separateDim[3] = number * pool_proj * height * width;


	host_channels[0] = one;
	host_channels[1] = three;
	host_channels[2] = five;
	host_channels[3] = pool_proj;

	separate_dstData.push_back(InnerLayers[0].getLayer("one")->dstData);
	separate_dstData.push_back(InnerLayers[1].getLayer("three")->dstData);
	separate_dstData.push_back(InnerLayers[2].getLayer("five")->dstData);
	separate_dstData.push_back(InnerLayers[3].getLayer("pool_proj")->dstData);

	separate_dstData.toGpu();

	dstData = NULL;
	dev_offset = NULL;
	dev_channels = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_offset, 4 * sizeof(int));
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dev_channels, 4 * sizeof(int));
	checkCudaErrors(cudaMemcpy(dev_offset, host_offset, 4 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_channels, host_channels, 4 * sizeof(int), cudaMemcpyHostToDevice));

	dim3 block(number, 4);
	dim3 thread(1024);
	MultiChannelsMerge<<<block,thread>>>(separate_dstData.devPoint, 
                                         dstData, 
                                         dev_channels, 
                                         dev_offset, 
                                         height, 
                                         channels);
	cudaThreadSynchronize();

	separate_dstData.vector_clear();
	MemoryMonitor::instanceObject()->freeGpuMemory(dev_offset);
	MemoryMonitor::instanceObject()->freeGpuMemory(dev_channels);
	return dstData;
}


/*split the delta*/
void Concat::split_DiffData(int index, float* diffData)
{

	separate_diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&separate_diffData, separateDim[index] * sizeof(float));

	if (0 == index)
	{
		InnerLayers[0].getLayer("one")->nextLayer->diffData = separate_diffData;

	} else if (1 == index)
	{
		InnerLayers[1].getLayer("three")->nextLayer->diffData = separate_diffData;
	}
	else if (2 == index)
	{
		InnerLayers[2].getLayer("five")->nextLayer->diffData = separate_diffData;
	} else
	{
		InnerLayers[3].getLayer("pool_proj")->nextLayer->diffData = separate_diffData;
	}


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
}


/*inception backwardPropagation*/
float* Concat::backwardSetup()
{
	prevDiff.push_back(InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[1].getLayer(InnerLayers[1].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[2].getLayer(InnerLayers[2].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[3].getLayer(InnerLayers[3].getLayersName(0))->diffData);
	prevDiff.toGpu();

	prev_number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->number;
	prev_channels = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->channels;
	prev_height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->height;
	prev_width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->width;

	size  = prev_number * prev_channels * prev_height * prev_width;
	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, size * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(diffData, size * sizeof(float));

    dim3 block(1);
    dim3 thread(1024);
    MultiArrayAdd<<<block, thread>>>(prevDiff.devPoint, 
                                     diffData,
                                     prev_number, 
                                     prev_channels, 
                                     prev_height, 
                                     prev_width);
    cudaThreadSynchronize();
    prevDiff.vector_clear();
	return diffData;

}
