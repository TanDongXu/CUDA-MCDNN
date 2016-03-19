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
    nextLayer.clear();
    prevLayer.clear();
	_inputImageDim = 0;
	_outputImageDim = 0;
	_inputAmount = 0;
	_outputAmount = 0;
	outputSize = 0;
}

