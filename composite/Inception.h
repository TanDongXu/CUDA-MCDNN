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
#include"../layers/convLayer.h"
#include"../layers/poolLayer.h"
#include"../layers/layersBase.h"
#include"../layers/ShareLayer.h"
#include"../tests/test_layer.h"



using namespace std;

class Inception
{
    public:
    typedef tuple<int, int, int, int, int, int, int, int, float, float>param_tuple;
    Inception(layersBase* prevLayer, int sign, float* rate, const param_tuple& args);
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float*& nextLayerDiffData, float Momemtum);

    /*get result*/
    float* getConcatData()
    {
        return dstData;
    }


    /*get delta*/
    float* getInceptionDiffData()
    {
        return diffData;
    }

    ~Inception()
    {
        delete share_Layer;
        delete concat;
        delete InnerLayers;
        delete Conv_one;
        delete Conv_three_reduce;
        delete Conv_three;
        delete Conv_five;
        delete Conv_five_reduce;
        delete Conv_pool_proj;
        delete max_pool;
    }

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

    private:
    Concat* concat;
    ShareLayer* share_Layer;
    Layers* InnerLayers;
    convLayer* Conv_one;
    convLayer* Conv_three_reduce;
    convLayer* Conv_three;
    convLayer* Conv_five;
    convLayer* Conv_five_reduce;
    convLayer* Conv_pool_proj;
    poolLayer* max_pool;

};



#endif /* INCEPTION_H_ */
