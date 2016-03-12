#include"Concat.h"

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

	offset = NULL;
	separate_channels = NULL;
	separate_dstData = NULL;


	oneDim = 0;
	threeDim = 0;
	fiveDim = 0;
	pool_projDim = 0;

	prev_oneDiff = NULL;
	prev_threeDiff = NULL;
	prev_fiveDiff = NULL;
	prev_projDiff = NULL;
	last_oneDiff = NULL;
	last_threeDiff = NULL;
	last_fiveDiff = NULL;
	last_projDiff = NULL;

	diffData = NULL;

	channels = one + three + five + pool_proj;
	InnerLayers = Inner_Layers;

	offset = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));
	separate_channels = (int*)MemoryMonitor::instanceObject()->cpuMallocMemory(4 * sizeof(int));

}


float*& Concat::forwardSetup()
{
	number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;


	offset[0] = 0;
	offset[1] = one * height * width;
	offset[2] = three * height * width;
	offset[3] = five * height * width;

	separate_channels[0] = one;
	separate_channels[1] = three;
	separate_channels[2] = five;
	separate_channels[3] = pool_proj;

	separate_dstData[0] = InnerLayers[0].getLayer("one")->dstData;
	separate_dstData[1] = InnerLayers[1].getLayer("three")->dstData;
	separate_dstData[2] = InnerLayers[2].getLayer("five")->dstData;
	separate_dstData[3] = InnerLayers[3].getLayer("pool_proj")->dstData;

	dstData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&dstData, number * channels * height * width * sizeof(float));

	dim3 block(number, 4);
	dim3 thread(1024);
	MultiChannelsMerge<<<block,thread>>>(separate_dstData, dstData, separate_channels, offset, height, number, channels);

	return dstData;
}



void Concat::split_DiffData(int index, float*& diffData)
{
	if(0 == index)
	{
		last_oneDiff = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&last_oneDiff, oneDim * sizeof(float));
		checkCudaErrors(cudaMemcpy(last_oneDiff, diffData, oneDim * sizeof(float), cudaMemcpyDeviceToDevice ));
		InnerLayers[0].getLayer("one")->nextLayer->diffData = last_oneDiff;
		//printf_DevParameter(number, one, 12, 12, last_oneDiff);

	}else if(1 == index)
	{
		last_threeDiff = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&last_threeDiff, threeDim * sizeof(float));
		checkCudaErrors(cudaMemcpy(last_threeDiff, diffData + oneDim, threeDim * sizeof(float), cudaMemcpyDeviceToDevice));
		InnerLayers[1].getLayer("three")->nextLayer->diffData = last_threeDiff;

	}else if(2 == index)
	{
		last_fiveDiff = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&last_fiveDiff, fiveDim * sizeof(float));
		checkCudaErrors(cudaMemcpy(last_fiveDiff, diffData + oneDim + threeDim, fiveDim * sizeof(float), cudaMemcpyDeviceToDevice));
		InnerLayers[2].getLayer("five")->nextLayer->diffData = last_fiveDiff;

	}else
	{
		last_projDiff = NULL;
        MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&last_projDiff, pool_projDim * sizeof(float));
        checkCudaErrors(cudaMemcpy(last_projDiff, diffData + oneDim + threeDim + fiveDim, pool_projDim * sizeof(float), cudaMemcpyDeviceToDevice));
        InnerLayers[3].getLayer("pool_proj")->nextLayer->diffData = last_projDiff;
        //printf_DevParameter(number, pool_proj, height, width, last_projDiff);
	}
}






float*& Concat::backwardSetup()
{
	/*the first share layer no need compute here*/
	prev_oneDiff = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->diffData;
	prev_threeDiff = InnerLayers[1].getLayer(InnerLayers[1].getLayersName(0))->diffData;
	prev_fiveDiff = InnerLayers[2].getLayer(InnerLayers[2].getLayersName(0))->diffData;
	prev_projDiff = InnerLayers[3].getLayer(InnerLayers[3].getLayersName(0))->diffData;

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
    MultiArrayAdd<<<block, thread>>>(prev_oneDiff, prev_threeDiff, prev_fiveDiff, prev_projDiff, diffData);

	return diffData;

}
