/*
 * ShareLayer.h
 *
 *  Created on: Mar 7, 2016
 *      Author: tdx
 */

#ifndef SHARELAYER_H_
#define SHARELAYER_H_

#include<tuple>
#include<string>
#include"layersBase.h"

class ShareLayer : public convLayerBase
{
public:
	ShareLayer(string name);
	ShareLayer(string name, convLayerBase* layer);
	void forwardPropagation(string train_or_test){};
	void backwardPropagation(float Momemtum){};
	void readWeight(FILE* file){};
	void saveWeight(FILE* file){};
	void Forward_cudaFree(){}
	void Backward_cudaFree(){}

	int getOutputSize()
	{
		return outputSize;
	}


private:
	int outputSize;

};




#endif /* SHARELAYER_H_ */
