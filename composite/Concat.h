/*
 * Concat.h
 *
 *  Created on: Mar 6, 2016
 *      Author: tdx
 */

#ifndef CONCAT_H_
#define CONCAT_H_

#include"../layers/layersBase.h"
#include"../common/MemoryMonitor.h"
#include"../common/checkError.h"
#include"../common/utility.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"
#include<tuple>

class Concat
{
public:
	typedef tuple<int, int, int, int>param_tuple;
	Concat(Layers*& Inner_Layers, const param_tuple& args);
	float*& forwardSetup();
	float*& backwardSetup();
	void split_DiffData(int index, float*& diffData);



private:
	int number;
	int channels;
    int height;
    int width;
    int prev_number;
    int prev_channels;
    int prev_height;
    int prev_width;
    int one;
    int three;
    int five;
    int pool_proj;
    int oneDim;
    int threeDim;
    int fiveDim;
    int pool_projDim;
	float* one_outResult;
	float* three_outResult;
	float* five_outResult;
	float* proj_outResult;
	float* outputResult;
	float* prev_oneDiff;
	float* prev_threeDiff;
	float* prev_fiveDiff;
	float* prev_projDiff;
	float* last_oneDiff;
	float* last_threeDiff;
	float* last_fiveDiff;
	float* last_projDiff;
	float* diffData;
	Layers *InnerLayers;


};



#endif /* CONCAT_H_ */
