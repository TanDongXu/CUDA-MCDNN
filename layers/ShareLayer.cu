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
	inputImageDim = 0;
	inputAmount = 0;
	outputSize = 0;
}

