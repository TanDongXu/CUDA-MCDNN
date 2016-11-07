#include"InceptionLayer.h"
#include"../config/config.h"

/*
 * Destructor
 * */
InceptionLayer::~InceptionLayer()
{
     delete inception;
};

/*
 * Get the outputSize
 * */
int InceptionLayer::getOutputSize()
{
    return outputSize;
}

/*
 * Inception layer constructor
 * */
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
    prevLayer.clear();
    nextLayer.clear();

    configInception* curConfig = (configInception*) config::instanceObjtce()->getLayersByName(_name);
    string prevLayerName = curConfig->_input;
    LayersBase* prev_Layer = (LayersBase*) Layers::instanceObject()->getLayer(prevLayerName);

    one = curConfig->_one;
    three = curConfig->_three;
    five = curConfig->_five;
    three_reduce = curConfig->_three_reduce;
    five_reduce = curConfig->_five_reduce;
    pool_proj = curConfig->_pool_proj;
    epsilon = curConfig->_init_w;
    lrate = curConfig->_lrate;
    lambda = curConfig->_weight_decay;

    inputAmount = prev_Layer->channels;
    inputImageDim = prev_Layer->height;
    number = prev_Layer->number;
    channels = one + three + five + pool_proj;
    height = prev_Layer->height;
    width = prev_Layer->width;
    outputSize = channels *  height * width;

    /*create inception*/
    inception = new Inception(prev_Layer, sign, &lrate,
                              Inception::param_tuple(one, three, five, three_reduce, five_reduce,
                                                     pool_proj, inputAmount, inputImageDim, epsilon, lambda));
}

/*
 * Inception layer forward propagation
 * */
void InceptionLayer::forwardPropagation(string train_or_test)
{
    srcData = prevLayer[0]->dstData;
    inception->forwardPropagation(train_or_test);
    dstData = inception->getConcatData();
}

/*
 * Inception layer backward propagation
 * */
void InceptionLayer::backwardPropagation(float Momentum)
{
    int nIndex = m_nCurBranchIndex;
    inception->backwardPropagation(nextLayer[nIndex]->diffData, Momentum);
    diffData = inception->getInceptionDiffData();
}

