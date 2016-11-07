#include"config.h"
#include<string>
#include<vector>
#include<string.h>
#include<iostream>
#include<map>
#include"opencv2/highgui.hpp"
using namespace std;

void config::deleteSpace()
{
    if(_configStr.empty())
    return;

    size_t pos1, pos2, e,t,n;
    while(1)
    {
        e = _configStr.find(' ');
        t = _configStr.find('\t');
        n = _configStr.find('\n');

        if(e == string::npos && t == string::npos && n == string::npos) break;

        if(e<t || t == string::npos) pos1 =e;
        else pos1 =t;

        if(n<pos1 || pos1 == string::npos) pos1=n;

        for(pos2 = pos1+1; pos2 < _configStr.size(); pos2++)
        {
            if(!(_configStr[pos2] == '\t' || _configStr[pos2] =='\n' ||_configStr[pos2] ==' '))
            break;
        }
        _configStr.erase(pos1, pos2 - pos1);
    }
}

void config::deleteComment()
{
    size_t pos1, pos2;
    while(1)
    {
        /**find the string first occur,0 begin*/
        pos1=_configStr.find("/");
        /*if no found return npos*/
        if(pos1 == string::npos)
        break;

        for(pos2 = pos1 +1; pos2 < _configStr.size(); pos2++)
        {
            if(_configStr[pos2] == '/')
            break;
        }
        _configStr.erase(pos1, pos2 - pos1+1);
    }
}

string config::read_to_string(string file_name)
{
    char* pBuf;
    FILE *fp = NULL;
    fp=fopen(file_name.c_str(),"r");
    if(!fp)
    {
        cout<<"read_to_string:Can not open the file"<<endl;
        exit(1);
    }
    /*move pointer to the end of the file*/
    fseek(fp,0,SEEK_END);
    /*The current position of the offset byte relative to the file head*/
    size_t lenght = ftell(fp);
    pBuf = new char[lenght];
    /*The pointer inside the file is re - pointing to the beginning of a stream*/
    rewind(fp);

    if(fread(pBuf,1,lenght,fp) != lenght)
    {
        cout<<"read_to_string: fread error"<<endl;
    }

    fclose(fp);
    string res = pBuf;
    return res;
}

int config::get_word_int(string&str, string name)
{
    size_t pos = str.find(name);
    int i= pos +1;
    int res = 1;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }

    string sub = str.substr(pos, i - pos + 1);
    /*note:';' have alread include in the string*/
    if(sub[sub.length()-1] ==';')
    {
        string content =sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = atoi(content.c_str());
    }
    str.erase(pos, i - pos + 1);
    return res;
}

float config::get_word_float(string&str, string name)
{
    size_t pos = str.find(name);
    int i = pos + 1;
    float res = 0.0f;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';')break;
        ++ i;
    }

    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length()-1] == ';')
    {
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = atof(content.c_str());
    }
    str.erase(pos ,i - pos + 1);
    return res;
}

void config::get_layers_config(string &str)
{
    vector<string> layers;

    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1)
    {
        if(head == str.length()) break ;
        if(str[head] == '['){
            tail = head + 1;
            while(1)
            {
                if(tail == str.length())break;
                if(str[tail] == ']') break;
                ++ tail;
            }
            string sub =str.substr(head, tail - head + 1);
            if(sub[sub.length()-1] == ']')
            {
                /*delete last ']'*/
                sub.erase(sub.begin()+sub.length()-1);
                /*delete first '['*/
                sub.erase(sub.begin());
                layers.push_back(sub);
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
    cout<<endl<<endl<<"...Read The Layers Configure :"<< endl;;
    for(int i=0; i< layers.size(); i++)
    {
        string type = get_word_type(layers[i], "LAYER");
        string name = get_word_type(layers[i], "NAME");
        string input = get_word_type(layers[i], "INPUT");
        string sub_input = get_word_type(layers[i], "SUB_INPUT");
        configBase* layer;
        if(string("CONV") == type)
        {
            int ks = get_word_int(layers[i], "KERNEL_SIZE");
            int ka = get_word_int(layers[i], "KERNEL_AMOUNT");
            int pad_h = get_word_int(layers[i], "PAD_H");
            int pad_w = get_word_int(layers[i], "PAD_W");
            int stride_h = get_word_int(layers[i], "STRIDE_H");
            int stride_w = get_word_int(layers[i], "STRIDE_W");

            float init_w = get_word_float(layers[i], "INIT_W");
            float lrate = get_word_float(layers[i], "LEARN_RATE");
            float weight_decay = get_word_float(layers[i], "WEIGHT_DECAY");

            layer = new configConv(type, name, input, sub_input, ks, pad_h, pad_w, stride_h,
                                   stride_w, ka, init_w, lrate, weight_decay);

            cout << endl;
            cout << "***********************Conv layer**********************"
            << endl;
            cout << "              NAME : " << name         << endl;
            cout << "             INPUT : " << input        << endl;
            cout << "         SUB_INPUT : " << sub_input    << endl;
            cout << "       KERNEL_SIZE : " << ks           << endl;
            cout << "     KERNEL_AMOUNT : " << ka           << endl;
            cout << "             PAD_H : " << pad_h        << endl;
            cout << "             PAD_W : " << pad_w        << endl;
            cout << "          STRIDE_H : " << stride_h     << endl;
            cout << "          STRIDE_W : " << stride_w     << endl;
            cout << "            INIT_W : " << init_w       << endl;
            cout << "        LEARN_RATE : " << lrate        << endl;
            cout << "      WEIGHT_DECAY : " << weight_decay << endl;

        }else if(string("POOLING") == type)
        {
            string poolType = get_word_type(layers[i], "POOLING_TYPE");
            m_poolMethod = new ConfigPoolMethod(poolType);
            int size = get_word_int(layers[i], "POOLDIM");
            int pad_h = get_word_int(layers[i], "PAD_H");
            int pad_w = get_word_int(layers[i], "PAD_W");
            int stride_h = get_word_int(layers[i], "STRIDE_H");
            int stride_w = get_word_int(layers[i], "STRIDE_W");

            layer = new configPooling(type, name, input, sub_input, size, pad_h, pad_w, stride_h,
                                      stride_w, m_poolMethod->getValue());

            cout << endl;
            cout << "***********************Pooling layer*******************"
            << endl;
            cout << "              NAME : " << name        << endl;
            cout << "             INPUT : " << input       << endl;
            cout << "         SUB_INPUT : " << sub_input   << endl;
            cout << "      POOLING_TYPE : " << poolType    << endl;
            cout << "           POOLDIM : " << size        << endl;
            cout << "             PAD_H : " << pad_h       << endl;
            cout << "             PAD_W : " << pad_w       << endl;
            cout << "          STRIDE_H : " << stride_h    << endl;
            cout << "          STRIDE_W : " << stride_w    << endl;

        }else if(string("HIDDEN") == type){
            int NumHidden = get_word_int(layers[i], "NUM_HIDDEN_NEURONS");
            float init_w = get_word_float(layers[i], "INIT_W");
            float lrate = get_word_float(layers[i], "LEARN_RATE");
            float weight_decay = get_word_float(layers[i], "WEIGHT_DECAY");

            layer = new configHidden(type, name, input, sub_input, NumHidden, init_w, lrate, weight_decay );

            cout << endl ;
            cout <<"***********************Hidden layer********************"<< endl;
            cout <<"              NAME : " << name          << endl;
            cout <<"             INPUT : " << input         << endl;
            cout <<"         SUB_INPUT : " << sub_input     << endl;
            cout <<"NUM_HIDDEN_NEURONS : "<< NumHidden << endl;
            cout <<"            INIT_W : " << init_w        << endl;
            cout <<"        LEARN_RATE : " << lrate         << endl;
            cout <<"      WEIGHT_DECAY : " << weight_decay  << endl;

        }else if(string("SOFTMAX") == type){
            int nclasses = get_word_int(layers[i], "NUM_CLASSES");
            float weight_decay = get_word_float(layers[i], "WEIGHT_DECAY");
            layer = new configSoftMax(type, name , input, sub_input, nclasses, weight_decay);

            cout<< endl ;
            cout<<"***********************SoftMax layer*******************"<< endl;

            cout <<"              NAME : " << name         << endl;
            cout <<"             INPUT : " << input        << endl;
            cout <<"         SUB_INPUT : " << sub_input    << endl;
            cout <<"       NUM_CLASSES : " << nclasses     << endl;
            cout <<"      WEIGHT_DECAY : " << weight_decay << endl;
            cout << endl<<endl;

        }else if(string("DATA") == type){
            layer = new configData(type ,name, input, sub_input);
            cout << endl ;
            cout <<"***********************Data layer**********************"<< endl;
            cout <<"              NAME : " << name  <<endl;

        }else if(type == string("ACTIVATION")){
            string non_linearity = get_word_type(layers[i], "NON_LINEARITY");
            m_nonLinearity = new configNonLinearity(non_linearity);
            layer = new configActivation(type, name, input, sub_input, m_nonLinearity->getValue());

            cout << endl;
            cout <<"********************Activation layer*******************"<< endl;
            cout <<"              NAME : " << name          << endl;
            cout <<"             INPUT : " << input         << endl;
            cout <<"        SUB_INPUT  : " << sub_input     << endl;
            cout <<"     NON_LINEARITY : " << non_linearity << endl;

        }else if(string("LRN") == type){
            unsigned lrnN = get_word_int(layers[i],"LRNN");
            float lrnAlpha = get_word_float(layers[i], "LRNALPHA");
            float lrnBeta = get_word_float(layers[i], "LRNBETA");

            layer = new configLRN(type, name, input, sub_input, lrnN, lrnAlpha, lrnBeta);

            cout << endl;
            cout << "***********************LRN layer**********************"<< endl;
            cout <<"               NAME : " << name         << endl;
            cout <<"              INPUT : " << input        << endl;
            cout <<"          SUB_INPUT : " << sub_input    << endl;
            cout <<"               LRNN : " << lrnN         << endl;
            cout <<"           LRNALPHA : " << lrnAlpha     << endl;
            cout <<"            LRNBETA : " << lrnBeta      << endl;

        }else if(string("INCEPTION") == type){
            int one = get_word_int(layers[i], "ONE");
            int three = get_word_int(layers[i], "THREE");
            int five = get_word_int(layers[i], "FIVE");
            int three_reduce = get_word_int(layers[i], "THREE_REDUCE");
            int five_reduce = get_word_int(layers[i], "FIVE_REDUCE");
            int pool_proj = get_word_int(layers[i], "POOL_PROJ");
            float init_w = get_word_float(layers[i], "INIT_W");
            float lrate = get_word_float(layers[i], "LEARN_RATE");
            float weight_decay = get_word_float(layers[i], "WEIGHT_DECAY");

            layer = new configInception(type, name, input, sub_input, one, three, five, three_reduce, five_reduce,
                                        pool_proj, init_w, lrate, weight_decay);
            cout << endl;
            cout <<"********************Inception layer*******************"<< endl;
            cout <<"              NAME : " << name         << endl;
            cout <<"             INPUT : " << input        << endl;
            cout <<"         SUB_INPUT : " << sub_input    << endl;
            cout <<"              ONE  : " << one          << endl;
            cout <<"             THREE : " << three        << endl;
            cout <<"              FIVE : " << five         << endl;
            cout <<"      THREE_REDUCE : " << three_reduce << endl;
            cout <<"       FIVE_REDUCE : " << five_reduce  << endl;
            cout <<"         POOL_PROJ : " << pool_proj    << endl;
            cout <<"            INIT_W : " << init_w       << endl;
            cout <<"        LEARN_RATE : " << lrate        << endl;
            cout <<"      WEIGHT_DECAY : " << weight_decay << endl;

        }else if(string("DROPOUT") == type){
            float rate = get_word_float(layers[i], "DROP_RATE");
            layer = new configDropOut(type, name, input, sub_input, rate);
            cout << endl;
            cout <<"*********************DropOut layer********************"<< endl;
            cout <<"              NAME : " << name         << endl;
            cout <<"             INPUT : " << input        << endl;
            cout <<"         SUB_INPUT : " << sub_input    << endl;
            cout <<"         DROP_RATE : " << rate         << endl;

        }else if(string("BRANCH") == type){
            vector<string> outputs = get_name_vector(layers[i], "OUTPUTS");
            layer = new configBranch(type, name, input, sub_input, outputs);
            cout << endl;
            cout <<"***********************Branch layer********************"<< endl;
            cout <<"              NAME : " << name         << endl;
            cout <<"             INPUT : " << input        << endl;
            cout <<"         SUB_INPUT : " << sub_input    << endl;
            cout <<"            OUTPUT : ";
            for(int n = 0; n < outputs.size(); n++)
            {
                cout<< outputs[n]<<" ";
            }
            cout<< endl;
        }
        insertLayerByName(name, layer);
        if(std::string("DATA") == type){
            _firstLayers=layer;
        }
        else{
            /*link the point*/
            _layerMaps[layer->_input]->_next.push_back(layer);
            _layerMaps[name]->_prev.push_back(_layerMaps[layer->_input] );
        }

        if(std::string("SOFTMAX") == type)
        {
            _lastLayer = layer;
        }
    }
}

void config::init(string path)
{
    /*read the string from config.txt*/
    _configStr = read_to_string(path);
    /*delete the comment and space*/
    deleteComment();
    deleteSpace();
    /*batch_size*/
    _batch_size = get_word_int(_configStr, "BATCH_SIZE");
    /*normalized_width*/
    _normalized_width = get_word_int(_configStr, "NORMALIZED_WIDTH");
    /*imageSize*/
    _imageSize = get_word_int(_configStr, "IMAGES_SIZE");
    /*channels*/
    _channels = get_word_int(_configStr, "CNANNELS");
    /*learn_rate*/
    // _lrate = get_word_float(_configStr, "LEARN_RATE");
    /*epochs*/
    _training_epochs = get_word_int(_configStr, "EPOCHS");
    /*iter_per_epo*/
    _iter_per_epo = get_word_int(_configStr, "ITER_PER_EPO");
    /*layers*/
    get_layers_config(_configStr);
}

/*get the type of layers*/
string config::get_word_type(string &str, string name)
{
    size_t pos = str.find(name);
    if(pos == str.npos) return "NULL";
    int i = pos + 1;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    string content ;
    if(sub[sub.length()-1] == ';')
    {
        content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
    }
    str.erase(pos, i - pos + 1);
    return content;
}

vector<string> config::get_name_vector(string &str, string name)
{
    vector<string> result;
    size_t pos = str.find(name);
    if(pos == str.npos) return result;
    int i = pos + 1;
    while(1)
    {
        if(i ==  str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos , i - pos + 1);
    string content;
    if(sub[sub.length()-1] == ';' )
    {
        content = sub.substr(name.length() + 1, sub.length() - name.length() -2);
    }

    str.substr(pos ,i - pos + 1);
    while(content.size())
    {
        size_t pos = content.find(",");
        if(pos == str.npos)
        {
            result.push_back(content);
            break;
        }else
        {
            result.push_back(content.substr(0, pos));
            content.erase(0, pos + 1);
        }
    }
    return result;
}
