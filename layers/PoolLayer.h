/*
* PoolLayer.h
*
*  Created on: Nov 28, 2015
*      Author: tdx
*/

#ifndef POOLLAYER_H_
#define POOLLAYER_H_

#include<tuple>
#include<string>
#include<cudnn.h>
#include<math.h>
#include"LayersBase.h"
#include"../common/utility.cuh"
#include"../config/config.h"
#include"../cuDNN_netWork.h"
#include"../tests/test_layer.h"

using namespace std;

/*
 * Class pool layer
 * */
class PoolLayer : public LayersBase
{
    public:
    typedef tuple<string, int, int, int, int, int, int, int> param_tuple;
    PoolLayer(string name);
    PoolLayer(string name, const param_tuple& agrs);
    PoolLayer(const PoolLayer* layer);
    PoolLayer(const configBase* templateConfig);
    ~PoolLayer();
    void forwardPropagation(string train_or_test);
    void backwardPropagation(float Momentum);
    void saveWeight(FILE*file){}
    void readWeight(FILE*file){}
    void createHandles();
    void destroyHandles();
    int getOutputSize();

    private:
    cudnnPoolingMode_t PoolingMode;
    ConfigPoolMethod* m_poolMethod;
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnTensorDescriptor_t dstTensorDesc;
    cudnnPoolingDescriptor_t poolingDesc;
    string pool_Type;
    int poolDim;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int outputSize;
    int prev_num;
    int prev_channels;
    int prev_height;
    int prev_width;
};

#endif /* POOLLAYER_H_ */
