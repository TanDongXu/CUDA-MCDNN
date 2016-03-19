#include"InceptionLayer.h"
#include"../config/config.h"


InceptionLayer::InceptionLayer(string name, int sign)
{
	_name = name;
	_inputName = " ";
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
    prevLayer.clear();
    nextLayer.clear();

    configInception* curConfig = (configInception*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
	convLayerBase* prev_Layer = (convLayerBase*) Layers::instanceObject()->getLayer(prevLayerName);

	one = curConfig->_one;
	three = curConfig->_three;
	five = curConfig->_five;
	three_reduce = curConfig->_three_reduce;
	five_reduce = curConfig->_five_reduce;
	pool_proj = curConfig->_pool_proj;
	epsilon = curConfig->_init_w;
	lrate = curConfig->_lrate;
	lambda = curConfig->_weight_decay;

	_inputAmount = prev_Layer->_outputAmount;
	_inputImageDim = prev_Layer->_outputImageDim;
	_outputAmount = one + three + five + pool_proj;
	_outputImageDim = _inputImageDim;
	outputSize = _outputAmount *  _outputImageDim * _outputImageDim;

	/*create inception*/
	inception = new Inception(prev_Layer, sign, &lrate,
			    Inception::param_tuple(one, three, five, three_reduce, five_reduce,
			    pool_proj, _inputAmount, _inputImageDim, epsilon, lambda));
}


void InceptionLayer::Forward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(srcData);
}


void InceptionLayer::forwardPropagation(string train_or_test)
{
	srcData = NULL;
	number = prevLayer[0]->number;
	channels = _outputAmount;
	height = prevLayer[0]->height;
	width = prevLayer[0]->width;
	srcData = prevLayer[0]->dstData;

	inception->forwardPropagation(train_or_test);
	dstData = inception->getConcatData();
}


void InceptionLayer::backwardPropagation(float Momentum)
{
	inception->backwardPropagation(nextLayer[0]->diffData, Momentum);
	diffData = inception->getInceptionDiffData();
}


void InceptionLayer::Backward_cudaFree()
{
	MemoryMonitor::instanceObject()->freeGpuMemory(dstData);
	MemoryMonitor::instanceObject()->freeGpuMemory(nextLayer[0]->diffData);
}

