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
	oneDim = 0;
	threeDim = 0;
	fiveDim = 0;
	pool_projDim = 0;
	one_outResult = NULL;
	three_outResult = NULL;
	five_outResult = NULL;
	proj_outResult = NULL;
	outputResult = NULL;
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

}


float*& Concat::forwardSetup()
{
	number = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->number;
	height = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->height;
	width = InnerLayers[0].getLayer(InnerLayers[0].getLayersName(0))->width;


	oneDim = number * one * height * width;
	threeDim = number * three * height * width;
	fiveDim = number * five * height * width;
	pool_projDim = number * pool_proj * height * width;

	one_outResult = InnerLayers[0].getLayer("one")->dstData;
	three_outResult = InnerLayers[1].getLayer("three")->dstData;
	five_outResult = InnerLayers[2].getLayer("five")->dstData;
	proj_outResult = InnerLayers[3].getLayer("pool_proj")->dstData;

	outputResult = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&outputResult, number * channels * height * width * sizeof(float));



	checkCudaErrors(cudaMemcpy(outputResult, one_outResult, oneDim * sizeof(float), cudaMemcpyDeviceToDevice ));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim, three_outResult, threeDim * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim + threeDim, five_outResult, fiveDim * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(outputResult + oneDim + threeDim + fiveDim, proj_outResult, pool_projDim * sizeof(float), cudaMemcpyDeviceToDevice));


	return outputResult;
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

	int size = prev_number * prev_channels * prev_height * prev_width;

	diffData = NULL;
	MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&diffData, size * sizeof(float));
    MemoryMonitor::instanceObject()->gpuMemoryMemset(diffData, size * sizeof(float));

	float alpha = 1.0;

	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
			                      size,
			                      &alpha,
			                      prev_oneDiff,
			                      1,
			                      diffData,
			                      1));

	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      prev_threeDiff,
				                      1,
				                      diffData,
				                      1));
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      prev_fiveDiff,
				                      1,
				                      diffData,
				                      1));
	checkCublasErrors(cublasSaxpy(cuDNN_netWork<float>::instanceObject()->GetcublasHandle(),
				                      size,
				                      &alpha,
				                      prev_projDiff,
				                      1,
				                      diffData,
				                      1));



	return diffData;

}
