/*
* config.h
*
*  Created on: Nov 24, 2015
*      Author: tdx
*/

#ifndef CONFIG_H_
#define CONFIG_H_

#include"../common/utility.cuh"
#include<string>
#include<vector>
#include<iostream>
#include<map>
using namespace std;

class configNonLinearity
{
    public:
    configNonLinearity(string method)
    {
        if(string("NL_SIGMOID") == method)
        {
            m_nonLinearity = ACTIVATION_SIGMOID;

        }else if(string("NL_TANH") == method)
        {
            m_nonLinearity = ACTIVATION_TANH;

        }else if(string("NL_RELU") == method)
        {
            m_nonLinearity = ACTIVATION_RELU;
        }else if(string("NL_LRELU") == method)
        {
            m_nonLinearity = ACTIVATION_LRELU;
        }
        else
        {
            cout<<"config: non_Linearity is not exist"<<endl;
            exit(0);
        }
    }

    int getValue()
    {
        return m_nonLinearity;
    }

    private:
    int m_nonLinearity;

};

class ConfigPoolMethod
{
    public:
    ConfigPoolMethod(string method)
    {
        if(string("POOL_MAX") == method)
        {
            m_poolMethod = POOLING_MAX;
        }else if(string("POOL_AVE_INCLUDE_PAD") == method)
        {
            m_poolMethod = POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        }else if(string("POOL_AVE_EXCLUDE_PAD") == method)
        {
            m_poolMethod = POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        }else
        {
            cout<<"config: POOL_METHOD is not exist"<<endl;
            exit(0);
        }
    }

    int getValue()
    {
        return m_poolMethod;
    }

    private:
    int m_poolMethod;
};

/*the base of config*/
class configBase
{
    public:
    string _type;
    string _name;
    string _input;
    string _sub_input;
    vector<configBase*>_next;
    vector<configBase*>_prev;
};

/*configure file in the config.txt*/
class config
{
    public:
    static config* instanceObjtce()
    {
        static config* conf =new config();
        return conf;
    }

    /*get batchSize*/
    int get_batchSize()
    {
        return _batch_size;
    }

    /*get normalizedWidth*/
    int get_normalizedWidth()
    {
        return _normalized_width;
    }

    /*get imageSize*/
    int get_imageSize()
    {
        return _imageSize;
    }

    /*get the total epochs*/
    int get_trainEpochs()
    {
        return _training_epochs;
    }

    /*iter of every epoch*/
    int get_iterPerEpo()
    {
        return _iter_per_epo;
    }

    /*get learn rate*/
    float get_lrate()
    {
        return _lrate;
    }

    /*get classification number*/
    int get_nclasses()
    {
        return _nclasses;
    }

    /*get channels*/

     int getChannels()
     {
         return _channels;
     }

    /*read the configure from file*/
    void initConfig(string path)
    {
          _path = path;
         init(_path);
    }

    configBase* getFirstLayers()
    {
        return _firstLayers;
    }

    configBase* getLastLayer(){
         return _lastLayer;
     }

    configBase* getLayersByName(std::string name)
    {
        if(_layerMaps.find(name) != _layerMaps.end()){
            return _layerMaps[name];
        }
        else{
            cout<<"config:layer "<<name<<" does not exist"<<endl;
            exit(0);
        }
    }

   //insert layer
   void insertLayerByName(std::string name, configBase* layer)
   {
        if(_layerMaps.find(name) == _layerMaps.end())
        {
            _layerMaps[name] = layer;
        }
        else
        {
            cout<<"config:layer "<<name<<" have already exist"<<endl;
            exit(0);
        }
   }

    size_t getLayersNum()
    {
        return _layerMaps.size();
    }

    config()
    {
        _lrate = 123456;
        _firstLayers = NULL;
        _lastLayer = NULL;
        m_poolMethod = NULL;
        m_nonLinearity = NULL;
    }

    private:
        int _batch_size;
        int _normalized_width;
        int _imageSize;
        int _training_epochs;
        int _iter_per_epo;
        float _lrate;
        int _nclasses;
        int _channels;
        string _configStr;
        string _path;
        map<string, configBase*> _layerMaps;
        configBase* _firstLayers;
        configBase* _lastLayer;
        configNonLinearity* m_nonLinearity;
        ConfigPoolMethod * m_poolMethod;

        void deleteSpace();
        void deleteComment();
        string get_word_type(string &str, string name);
        float get_word_float(string &str,string name);
        int get_word_int(string &str, string name);
        string read_to_string(string file_name);
        void get_layers_config(string&str);
        vector<string> get_name_vector(string &str, string name);
        void init(string path);
};

/*Each layer config*/
/*data layer config */
class configData : public configBase
{
    public:
    configData(string type, string name, string input, string sub_input)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
    }
};

/*conv layer config*/
class configConv : public configBase
{
    public:
    configConv(string type, string name, string input, string sub_input,
               int kernelSize, int pad_h, int pad_w, int stride_h, int stride_w,
               int kernelAmount, float init_w, float lrate, float weight_decay)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _kernelSize = kernelSize;
        _pad_h = pad_h;
        _pad_w = pad_w;
        _stride_h = stride_h;
        _stride_w = stride_w;
        _kernelAmount = kernelAmount;
        _init_w = init_w;
        _lrate = lrate;
        _weight_decay = weight_decay;

    }

    int _kernelSize;
    int _pad_h;
    int _pad_w;
    int _stride_h;
    int _stride_w;
    int _kernelAmount;
    float _weight_decay;
    float _init_w;
    float _lrate;
};


/*pool layer config*/
class configPooling : public configBase
{
    public:
    configPooling(string type, string name, string input, string sub_input, int size,
                  int pad_h, int pad_w, int stride_h, int stride_w, int poolType)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _size = size;
        _pad_h = pad_h;
        _pad_w = pad_w;
        _stride_h = stride_h;
        _stride_w = stride_w;
        _poolType = poolType;
    }

    int _size;
    int _pad_h;
    int _pad_w;
    int _stride_h;
    int _stride_w;
    int _poolType;

};

/*inception layer config*/
class configInception : public configBase
{
    public:
    configInception(string type, string name, string input, string sub_input,
                    int one, int three, int five, int three_reduce, int five_reduce,
                    int pool_proj, float init_w, float lrate, float weight_decay)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _one = one;
        _three = three;
        _five = five;
        _three_reduce = three_reduce;
        _five_reduce = five_reduce;
        _pool_proj = pool_proj;
        _init_w = init_w;
        _weight_decay = weight_decay;
        _lrate = lrate;
    }

    int _one;
    int _three;
    int _five;
    int _three_reduce;
    int _five_reduce;
    int _pool_proj;
    float _weight_decay;
    float _init_w;
    float _lrate;
};

/*branch layer config*/
class configBranch : public configBase
{
    public:
    configBranch(string type, string name, string input, string sub_input, vector<string>outputs)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        m_outputs = outputs;
    }
    vector<string> m_outputs;
};

/*hidden layer config*/
class configHidden :public configBase
{
    public:
    configHidden(string type, string name, string input, string sub_input,
                 int NumHiddenNeurons, float init_w, float lrate, float weight_decay)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _NumHiddenNeurons = NumHiddenNeurons;
        _init_w = init_w;
        _lrate = lrate;
        _weight_decay = weight_decay;
    }

    int _NumHiddenNeurons;
    float _init_w;
    float _lrate;
    float _weight_decay;
};

/*dropout layer config*/
class configDropOut : public configBase
{
    public:
    configDropOut(string type, string name, string input, string sub_input, float rate){
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        dropOut_rate = rate;
    }

    public:
    float dropOut_rate;
};

/*activation layer config*/
class configActivation : public configBase
{
    public:
    configActivation(string type, string name, string input, string sub_input, int non_linearity)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _non_linearity = non_linearity; 
    }
    int _non_linearity;
};

/*LRN layer config*/
class configLRN : public configBase
{
    public:
    configLRN(string type, string name, string input, string sub_input, unsigned lrnN, float lrnAlpha,float lrnBeta)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _lrnN = lrnN;
        _lrnAlpha = lrnAlpha;
        _lrnBeta = lrnBeta;
    }

    unsigned _lrnN;
    float _lrnAlpha;
    float _lrnBeta;
};

/*softmax layer config*/
class configSoftMax : public configBase
{
    public:
    configSoftMax(string type, string name, string input, string sub_input, int nclasses, float weight_decay)
    {
        _type = type;
        _name = name;
        _input = input;
        _sub_input = sub_input;
        _nclasses = nclasses;
        _weight_decay = weight_decay;
    }

    int _nclasses;
    float _weight_decay;
};


#endif /* CONFIG_H_ */
