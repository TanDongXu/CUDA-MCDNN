#include"Concat.h"


void Concat::concatInit()
{
	host_offset = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
	separateDim = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
	host_channels = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));

	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&lastDiff, 4 * sizeof(float*));

}


Concat::Concat(Layers*& Inner_Layers, const param_tuple& args)
{
	std::tie(one, three, five, pool_proj) = args;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	prev_number = 0;
	prev_channels = 0;
	prev_height = 0;
	prev_width = 0;
	host_offset = NULL;
	dev_offset = NULL;
	separateDim = NULL;
	host_channels = NULL;
	dev_channels = NULL;
	lastDiff = NULL;
	diffData = NULL;
	dstData = NULL;

	channels = one + three + five + pool_proj;
	InnerLayers = Inner_Layers;

	this->concatInit();

}


float* Concat::forwardSetup()
{
	number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;


	host_offset[0] = 0;
	host_offset[1] = one * height * width;
	host_offset[2] = (one + three) * height * width ;
	host_offset[3] = (one + three + five) * height * width;

	//use in next function
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
	MultiChannelsMerge<<<block,thread>>>(separate_dstData.hostPoint, dstData, dev_channels, dev_offset, height, channels);
	cudaThreadSynchronize();

	cout<<number<<" "<<channels<<" "<<height<<" "<<width<<endl;
	printf_DevParameter(number,channels,height,width,dstData);
	cout<<"asdfadf"<<endl;

	separate_dstData.vector_clear();
	MemoryMonitor::instanceObject()->freeGpuMemory(dev_offset);
	MemoryMonitor::instanceObject()->freeGpuMemory(dev_channels);

	return dstData;
}



void Concat::split_DiffData(int index, float*& diffData)
{
	if(0 == index)
	{
		lastDiff[0] = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&lastDiff[0], separateDim[0] * sizeof(float));
		checkCudaErrors(cudaMemcpy(lastDiff[0], diffData, separateDim[0] * sizeof(float), cudaMemcpyDeviceToDevice ));
		InnerLayers[0].getLayer("one")->nextLayer->diffData = lastDiff[0];
		//printf_DevParameter(number, one, 12, 12, last_oneDiff);

	}else if(1 == index)
	{
		lastDiff[1] = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&lastDiff[1], separateDim[1] * sizeof(float));
		checkCudaErrors(cudaMemcpy(lastDiff[1], diffData + separateDim[0], separateDim[1] * sizeof(float), cudaMemcpyDeviceToDevice));
		InnerLayers[1].getLayer("three")->nextLayer->diffData = lastDiff[1];

	}else if(2 == index)
	{
		lastDiff[2] = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&lastDiff[2], separateDim[2] * sizeof(float));
		checkCudaErrors(cudaMemcpy(lastDiff[2], diffData + separateDim[0] + separateDim[1], separateDim[2] * sizeof(float), cudaMemcpyDeviceToDevice));
		InnerLayers[2].getLayer("five")->nextLayer->diffData = lastDiff[2];

	}else
	{
		lastDiff[3] = NULL;
        MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&lastDiff[3], separateDim[3] * sizeof(float));
        checkCudaErrors(cudaMemcpy(lastDiff[3], diffData + separateDim[0] + separateDim[1] + separateDim[2], separateDim[3] * sizeof(float), cudaMemcpyDeviceToDevice));
        InnerLayers[3].getLayer("pool_proj")->nextLayer->diffData = lastDiff[3];
        //printf_DevParameter(number, pool_proj, height, width, last_projDiff);
	}
}






float* Concat::backwardSetup()
{
	/*the first share layer no need compute here*/
	prevDiff.push_back(InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[1].getLayer(InnerLayers[1].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[2].getLayer(InnerLayers[2].getLayersName(0))->diffData);
	prevDiff.push_back(InnerLayers[3].getLayer(InnerLayers[3].getLayersName(0))->diffData);
	prevDiff.toGpu();

	prev_number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->number;
	prev_channels = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->channels;
	prev_height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->height;
	prev_width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->prevLayer->width;

	int size  = prev_number * prev_channels * prev_height * prev_width;
	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, size * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(diffData, size * sizeof(float));

    dim3 block(prev_number);
    dim3 thread(prev_channels * prev_height * prev_width);
    MultiArrayAdd<<<block, thread>>>(prevDiff.hostPoint[0], prevDiff.hostPoint[1], prevDiff.hostPoint[2], prevDiff.hostPoint[3], diffData);

    prevDiff.vector_clear();
	return diffData;

}
