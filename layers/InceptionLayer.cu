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
	prevLayer = NULL;
	nextLayer = NULL;

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
	inception = new Inception(prev_Layer, sign,
			    Inception::param_tuple(one, three, five, three_reduce, five_reduce,
			    pool_proj, _inputAmount, _inputImageDim, epsilon, lrate, lambda));

}

void InceptionLayer::forwardPropagation(string train_or_test)
{
	number = prevLayer->number;
	channels = _outputAmount;
	height = prevLayer->height;
	width = prevLayer->width;

	inception->forwardPropagation(train_or_test);
	dstData = inception->getConcatData();
}

void InceptionLayer::backwardPropagation(float Momentum)
{
	inception->backwardPropagation(nextLayer->diffData, Momentum);
	cout<<"inceptionLayer pos"<<endl;
	diffData = inception->getInceptionDiffData();

}

void InceptionLayer::saveWeight(FILE* file)
{

}

void InceptionLayer::readWeight(FILE* file)
{

}
