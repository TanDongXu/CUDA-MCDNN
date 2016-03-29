/*
 * NodeFission.h
 *
 *  Created on: Mar 24, 2016
 *      Author: tdx
 */

#ifndef NODEFISSION_H_
#define NODEFISSION_H_

#include"../layers/layersBase.h"
#include"../common/utility.cuh"
#include"../config/config.h"
#include"../layers/dataLayer.h"
#include"../layers/convLayer.h"
#include"../layers/activationLayer.h"
#include"../layers/dropOutLayer.h"
#include"../layers/hiddenLayer.h"
#include"../layers/LRNLayer.h"
#include"../layers/poolLayer.h"
#include"../layers/softMaxLayer.h"



void NodeFission(layersBase* splitNode, layersBase* nextNode);


//static factory
class FissionFactory
{
public:
	static FissionFactory* instanceObject(){

		static FissionFactory* factory = new FissionFactory();
		return factory;
	}

	static layersBase* createLayer(layersBase* node)
	{

		string nodeType = config::instanceObjtce()->getLayersByName(node->_name)->_type;

		if(nodeType == "DATA")
		{
			return new dataLayer((dataLayer*)node);
		}else if(nodeType == "CONV")
		{
			return new convLayer((convLayer*)node);
		}else if(nodeType == "POOLING")
		{
			return new poolLayer((poolLayer*)node);;

		}else if(nodeType == "HIDDEN")
		{
			return new hiddenLayer((hiddenLayer*)node);
		}else if(nodeType == "SOFTMAX")
		{
			return new softMaxLayer((softMaxLayer*)node);
		}else if(nodeType == "ACTIVATION")
		{
			return new activationLayer((activationLayer*)node);
		}else if(nodeType == "LRN")
		{
			return new LRNLayer((LRNLayer*)node);
		}else if(nodeType == "DROPOUT")
		{
			return new dropOutLayer((dropOutLayer*)node);
		}else
		{
			cout<< "NodeFisson: Layer not exist" << endl;
			exit(0);
		}
	}


	static configBase* createConfig(layersBase* node)
	{
		configBase* curConfig = config::instanceObjtce()->getLayersByName(node->_name);
		string nodeType = config::instanceObjtce()->getLayersByName(node->_name)->_type;

		if (nodeType == "DATA")
		{
			static int idx = 0;
			configData* tmp = (configData*) curConfig;
			configData* config = new configData( *tmp );
			config->_name = curConfig->_name + int_to_string(idx);
			idx ++;
			return config;

		} else if (nodeType == "CONV")
		{
			static int idx = 0;
			configConv* tmp  = (configConv*)curConfig;
			configConv* config = new configConv(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "POOLING")
		{
			static int idx = 0;
			configPooling* tmp = (configPooling*)curConfig;
			configPooling* config = new configPooling(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "HIDDEN")
		{
			static int idx = 0;
			configHidden* tmp = (configHidden*)curConfig;
			configHidden* config =new configHidden (*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "SOFTMAX")
		{
			static int idx = 0;
			configSoftMax* tmp = (configSoftMax*)curConfig;
			configSoftMax* config = new configSoftMax(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "ACTIVATION")
		{
			static int idx = 0;
			configActivation* tmp = (configActivation*)curConfig;
			configActivation* config = new configActivation(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "LRN")
		{
			static int idx = 0;
			configLRN* tmp = (configLRN*)curConfig;
			configLRN* config = new configLRN(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;

		} else if (nodeType == "DROPOUT")
		{
			static int idx = 0;
			configDropOut* tmp = (configDropOut*)curConfig;
			configDropOut* config = new configDropOut(*tmp);
			config->_name = curConfig->_name + int_to_string(idx);
			idx++;
			return config;
		}else
		{
			cout<< "NodeFission: Config not exist" << endl;
			exit(0);
		}
	}

};


#endif /* NODEFISSION_H_ */
