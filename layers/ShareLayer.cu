#include"ShareLayer.h"

ShareLayer::ShareLayer(string name)
{
	_name = name;
	_inputName = " ";
	srcData = NULL;
	dstData = NULL;
	diffData = NULL;
	number = 0;
	channels = 0;
	height = 0;
	width = 0;
	lrate = 0.0f;
	nextLayer = NULL;
	prevLayer = NULL;
	//InputLayer = NULL;
	_inputImageDim = 0;
	_outputImageDim = 0;
	_inputAmount = 0;
	_outputAmount = 0;
	outputSize = 0;
}

//ShareLayer::ShareLayer(string name, convLayerBase* layer)
//{
//	_name = name;
//	_inputName = " ";
//	lrate = 0.0f;
//	srcData = NULL;
//	dstData = NULL;
//	diffData = NULL;
//	nextLayer = NULL;
//	prevLayer = NULL;
//
//	//InputLayer = layer;
//
//	number = layer->number;
//	channels = layer->channels;
//	height = layer->height;
//	width = layer->width;
//
//	_inputImageDim = layer->_inputImageDim;
//	_outputImageDim = layer->_outputImageDim ;
//	_inputAmount = layer->_inputAmount;
//	_outputAmount = layer->_outputAmount;
//	outputSize = layer->getOutputSize();
//
//
//}
