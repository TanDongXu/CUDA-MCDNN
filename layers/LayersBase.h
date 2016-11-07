/*
* LayersBase.h
*
*  Created on: Nov 23, 2015
*      Author: tdx
*/

#ifndef LAYERSBASE_H_
#define LAYERSBASE_H_

#include"../common/cuMatrix.h"
#include<string>
#include<map>
#include<vector>
#include"math.h"
using namespace std;

/*
 * Class LayersBase
 * */
class LayersBase
{
    public:
    LayersBase():m_nCurBranchIndex(0), m_fReduceRate(0), lrate( 123456 ){}
    virtual void forwardPropagation(string train_or_test) = 0;
    virtual void backwardPropagation(float Momentum) = 0;
    virtual int getOutputSize() = 0;
    virtual void saveWeight(FILE*file) = 0;
    virtual void readWeight(FILE*file) = 0;

    void setCurBranchIndex(int nIndex = 0);
    void adjust_learnRate(int index, double lr_gamma, double lr_power);
    void insertPrevLayer(LayersBase* layer);
    void insertNextlayer(LayersBase* layer);
    void rateReduce();
    void setRateReduce( float fReduce);
    float getRateReduce();

    public:
    string _name;
    string _inputName;
    int number;
    int channels;
    int height;
    int width;
    int inputImageDim;
    int inputAmount;
    int m_nCurBranchIndex;
    float lrate;
    float m_fReduceRate;
    float *diffData;
    float *srcData , *dstData;
    vector<LayersBase*>prevLayer;
    vector<LayersBase*>nextLayer;
};

/*
 * Class Layers
 * */
class Layers
{
    public:

    static Layers* instanceObject()
    {
        static Layers* layers = new Layers();
        return layers;
    }

    /*get layer by name*/
    LayersBase * getLayer(string name);
    /*linear store the layers by name*/
    void storLayers(string name, LayersBase* layer);
    void storLayers(string prev_name, string name, LayersBase* layer);
    /*store the layers name*/
    void storLayersName(string);
    /*get layers name by index*/
    string getLayersName(int index);
    /*get layer num*/
    int getLayersNum()
    {
        return _layersMaps.size();
    }
    bool hasLayer(string name);

    private:
    map<string,LayersBase*> _layersMaps;
    vector<string> _layersName;
};


#endif /* LAYERSBASE_H_ */
