/*
* InceptionLayer.h
*
*  Created on: Mar 5, 2016
*      Author: tdx
*/

#ifndef INCEPTIONLAYER_H_
#define INCEPTIONLAYER_H_

#include<tuple>
#include"LayersBase.h"
#include"ConvLayer.h"
#include"PoolLayer.h"
#include"../composite/Inception.h"

/*
 * Class Inception layer
 * */
class InceptionLayer : public LayersBase
{
    public:
    InceptionLayer(string name, int sign);
    ~InceptionLayer();
    int getOutputSize();
    void ReShape(){}
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE* file){};
    void readWeight(FILE* file){};

    private:
    int one;
    int three;
    int five;
    int three_reduce;
    int five_reduce;
    int pool_proj;
    int outputSize;
    float lambda;
    float epsilon;
    Inception * inception;

};

#endif /* INCEPTIONLAYER_H_ */
