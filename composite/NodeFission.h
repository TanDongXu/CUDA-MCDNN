/*
* NodeFission.h
*
*  Created on: Mar 24, 2016
*      Author: tdx
*/

#ifndef NODEFISSION_H_
#define NODEFISSION_H_

#include"../layers/LayersBase.h"
#include"../common/utility.cuh"
#include"../config/config.h"
#include"../layers/DataLayer.h"
#include"../layers/ConvLayer.h"
#include"../layers/ActivationLayer.h"
#include"../layers/DropOutLayer.h"
#include"../layers/HiddenLayer.h"
#include"../layers/LRNLayer.h"
#include"../layers/PoolLayer.h"
#include"../layers/SoftMaxLayer.h"



void NodeFission(LayersBase* splitNode, LayersBase* nextNode);
void softmaxFission(LayersBase* splitNode);


//static factory
class FissionFactory
{
    public:
    static FissionFactory* instanceObject(){

        static FissionFactory* factory = new FissionFactory();
        return factory;
    }

    static LayersBase* createLayer(LayersBase* node)
    {

        string nodeType = config::instanceObjtce()->getLayersByName(node->_name)->_type;

        if(string("DATA") == nodeType)
        {
            return new DataLayer((DataLayer*)node);
        }else if(string("CONV") == nodeType)
        {
            return new ConvLayer((ConvLayer*)node);
        }else if(string("POOLING") == nodeType)
        {
            return new PoolLayer((PoolLayer*)node);;

        }else if(string("HIDDEN") == nodeType)
        {
            return new HiddenLayer((HiddenLayer*)node);
        }else if(string("SOFTMAX") == nodeType)
        {
            return new SoftMaxLayer((SoftMaxLayer*)node);
        }else if(string("ACTIVATION") == nodeType)
        {
            return new ActivationLayer((ActivationLayer*)node);
        }else if(string("LRN") == nodeType)
        {
            return new LRNLayer((LRNLayer*)node);
        }else if(string("DROPOUT") == nodeType)
        {
            return new DropOutLayer((DropOutLayer*)node);
        }else
        {
            cout<< "NodeFisson: Layer not exist" << endl;
            exit(0);
        }
    }


    static configBase* createConfig(LayersBase* node)
    {
        configBase* curConfig = config::instanceObjtce()->getLayersByName(node->_name);
        string nodeType = config::instanceObjtce()->getLayersByName(node->_name)->_type;

        if (string("DATA") == nodeType)
        {
            static int idx = 0;
            configData* tmp = (configData*) curConfig;
            configData* config = new configData( *tmp );
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx ++;
            return config;

        } else if (string("CONV") == nodeType)
        {
            static int idx = 0;
            configConv* tmp  = (configConv*)curConfig;
            configConv* config = new configConv(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("POOLING") == nodeType)
        {
            static int idx = 0;
            configPooling* tmp = (configPooling*)curConfig;
            configPooling* config = new configPooling(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("HIDDEN") == nodeType)
        {
            static int idx = 0;
            configHidden* tmp = (configHidden*)curConfig;
            configHidden* config =new configHidden (*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("SOFTMAX") == nodeType)
        {
            static int idx = 0;
            configSoftMax* tmp = (configSoftMax*)curConfig;
            configSoftMax* config = new configSoftMax(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("ACTIVATION") == nodeType)
        {
            static int idx = 0;
            configActivation* tmp = (configActivation*)curConfig;
            configActivation* config = new configActivation(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("LRN") == nodeType){
            static int idx = 0;
            configLRN* tmp = (configLRN*)curConfig;
            configLRN* config = new configLRN(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;

        } else if (string("DROPOUT") == nodeType){
            static int idx = 0;
            configDropOut* tmp = (configDropOut*)curConfig;
            configDropOut* config = new configDropOut(*tmp);
            config->_name = curConfig->_name + string("_") + int_to_string(idx);
            idx++;
            return config;
        }else{
            cout<< "NodeFission: Config not exist" << endl;
            exit(0);
        }
}

};


#endif /* NODEFISSION_H_ */
