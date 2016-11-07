/*
* Inception.h
*
*  Created on: Mar 6, 2016
*      Author: tdx
*/

#ifndef INCEPTION_H_
#define INCEPTION_H_


#include<tuple>
#include<time.h>
#include"Concat.h"
#include"../layers/ConvLayer.h"
#include"../layers/PoolLayer.h"
#include"../layers/LayersBase.h"
#include"../layers/ShareLayer.h"
#include"../tests/test_layer.h"



using namespace std;

class Inception
{
    public:
    typedef tuple<int, int, int, int, int, int, int, int, float, float>param_tuple;
    Inception(LayersBase* prevLayer, int sign, float* rate, const param_tuple& args);
    ~Inception();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float*& nextLayerDiffData, float Momemtum);
    /*get result*/
    float* getConcatData();
    /*get delta*/
    float* getInceptionDiffData();

    private:
    int one;
    int three;
    int five;
    int three_reduce;
    int five_reduce;
    int pool_proj;
    int inputAmount;
    int inputImageDim;
    float epsilon;
    float lambda;
    float* lrate;
    float* dstData;
    float* diffData;
    Concat* concat;
    ShareLayer* share_Layer;
    Layers* InnerLayers;
    ConvLayer* Conv_one;
    ConvLayer* Conv_three_reduce;
    ConvLayer* Conv_three;
    ConvLayer* Conv_five;
    ConvLayer* Conv_five_reduce;
    ConvLayer* Conv_pool_proj;
    PoolLayer* max_pool;

};



#endif /* INCEPTION_H_ */
