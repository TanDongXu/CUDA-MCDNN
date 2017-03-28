#include"NodeFission.h"
#include"../config/config.h"
#include"../layers/LayersBase.h"

//node fission
void NodeFission(LayersBase* splitNode, LayersBase* nextNode) {
    vector<LayersBase*>::iterator l_iter;
    vector<configBase*>::iterator c_iter;

    //modify config
    configBase* configFiss = FissionFactory::instanceObject()->createConfig(splitNode);

    configBase* curConfig = config::instanceObjtce()->getLayersByName(splitNode->_name);
    configBase* prevConfig = config::instanceObjtce()->getLayersByName(splitNode->_inputName);
    configBase* nextConfig = config::instanceObjtce()->getLayersByName(nextNode->_name);
    //necessary
    configFiss->_prev.clear();
    configFiss->_next.clear();

    prevConfig->_next.push_back(configFiss);
    configFiss->_prev.push_back(prevConfig);
    configFiss->_next.push_back(nextConfig);
    nextConfig->_prev.push_back(configFiss);

    config::instanceObjtce()->insertLayerByName(configFiss->_name, configFiss);

    for (c_iter = curConfig->_next.begin(); c_iter != curConfig->_next.end(); c_iter++) {
        if ((*c_iter)->_name == (nextConfig->_name)) {
            curConfig->_next.erase(c_iter);
            break;
        }
    }

    for (c_iter = nextConfig->_prev.begin(); c_iter != nextConfig->_prev.end(); c_iter++) {
        if ((*c_iter)->_name == (curConfig->_name)) {
            nextConfig->_prev.erase(c_iter);
            break;
        }
    }

    //这还要修改next的输入名字,因为只是删除next prev的指针,不用深度优先会出现错误
    nextConfig->_input = configFiss->_name;

    //modify Layers
    LayersBase* layerFiss = FissionFactory::instanceObject()->createLayer(splitNode);
    layerFiss->nextLayer.clear();
    layerFiss->prevLayer.clear();
    //layerFiss
    LayersBase* prevLayer = Layers::instanceObject()->getLayer(splitNode->_inputName);
    //prev
    Layers::instanceObject()->storLayers(prevLayer->_name, layerFiss->_name,layerFiss);
    //next
    layerFiss->insertNextlayer(nextNode);
    //splitNode
    //delete next
    for (l_iter = splitNode->nextLayer.begin(); l_iter != splitNode->nextLayer.end(); l_iter++) {
        if ((*l_iter)->_name == nextNode->_name) {
            splitNode->nextLayer.erase(l_iter);
            break;
        }
    }
    //delete prev
    for (l_iter = nextNode->prevLayer.begin(); l_iter != nextNode->prevLayer.end(); l_iter++) 
    {
        if ((*l_iter)->_name == (splitNode->_name)) {
            nextNode->prevLayer.erase(l_iter);
            break;
        }
    }
    //修改名字
    nextNode->_inputName = layerFiss->_name;
    //insert one node
    nextNode->insertPrevLayer(layerFiss);

}

//node fission for dynamic_generate model
void nodeGenerate(config* endConfig) 
{
    vector<LayersBase*>::iterator l_iter;
    vector<configBase*>::iterator c_iter;

    configBase* templateConfig = endConfig->getFirstLayers();
    configBase* curConfig = config::instanceObjtce()->getFirstLayers();
    configBase* newConfig_prev = NULL;
    configBase* newConfig_next = NULL;
    //find the first different node 
    while(0 != templateConfig->_next.size() && 0 != curConfig->_next.size())
    {
        if(templateConfig->_type == "SOFTMAX" || curConfig->_type == "SOFTMAX") 
            LOG(FATAL) << " Config Table Error, SoftMax not finally layer.";
        if(templateConfig->_name == curConfig->_name)
        {
            newConfig_prev = curConfig;
            templateConfig = templateConfig->_next[0];
            curConfig = curConfig->_next[0];
        }else
        {
            //this time, templateConfig point to the first different node
            newConfig_next = curConfig;
            break;
        }
    }

    cout<<"enter-1"<<endl;
    //make sure the point is not null 
    if(NULL == newConfig_prev && NULL == newConfig_next) return;
    //modify config
    configBase* newConfig = DynamicFactory::instanceObject()->createConfig(templateConfig);
    //necessary
    newConfig->_prev.clear();
    newConfig->_next.clear();

    cout<<newConfig->_name<<" "<<newConfig->_type<<" "<<newConfig->_input<<" "<<newConfig->_prev.size()<<" "<< newConfig->_next.size()<<endl;

    newConfig_prev->_next.push_back(newConfig);
    newConfig->_prev.push_back(newConfig_prev);
    newConfig->_next.push_back(newConfig_next);
    newConfig_next->_prev.push_back(newConfig);

    cout<<newConfig->_name<<" "<<newConfig->_type<<" "<<newConfig->_input<<" "<<newConfig->_prev[0]->_name<<" "<< newConfig->_next[0]->_name<<endl;
    //insert into origin config table
    cout<<"no hidden-1"<<endl;
    config::instanceObjtce()->insertLayerByName(newConfig->_name, newConfig);
    cout<<"no hidden"<<endl;
    // prev layer delete origin next layer point
    for (c_iter = newConfig_prev->_next.begin(); c_iter != newConfig_prev->_next.end(); c_iter++) 
    {
        if ((*c_iter)->_name == (newConfig_next->_name)) {
            newConfig_prev->_next.erase(c_iter);
            break;
        }
    }
    // next layer delete origin prev layer point
    for (c_iter = newConfig_next->_prev.begin(); c_iter != newConfig_next->_prev.end(); c_iter++) {
        if ((*c_iter)->_name == (newConfig_prev->_name)) {
            newConfig_next->_prev.erase(c_iter);
            break;
        }
    }
    
    cout<<newConfig_next->_input<<endl;
    //modify nextConfig input name
    newConfig_next->_input = newConfig->_name;
    cout<<newConfig_next->_input<<endl;


    //modify Layers by new config node
    LayersBase* newLayer = DynamicFactory::instanceObject()->createLayer(newConfig);
    newLayer->nextLayer.clear();
    newLayer->prevLayer.clear();
    //prev layer
    LayersBase* newLayer_prev = Layers::instanceObject()->getLayer(newConfig->_input);
    //next layer
    LayersBase* newLayer_next = Layers::instanceObject()->getLayer(newConfig_next->_name);
    // new node insert next layer
    newLayer->insertNextlayer(newLayer_next);
    //insert into layer table
    Layers::instanceObject()->storLayers(newLayer_prev->_name, newLayer->_name, newLayer);
    //delete next
    for (l_iter = newLayer_prev->nextLayer.begin(); l_iter != newLayer_prev->nextLayer.end(); l_iter++) 
    {
        if ((*l_iter)->_name == newLayer_next->_name) {
            newLayer_prev->nextLayer.erase(l_iter);
            break;
        }
    }
    //delete prev
    for (l_iter = newLayer_next->prevLayer.begin(); l_iter != newLayer_next->prevLayer.end(); l_iter++) 
    {
        if ((*l_iter)->_name == (newLayer_prev->_name)) 
        {
            newLayer_next->prevLayer.erase(l_iter);
            break;
        }
    }

    //must: modify nextlayer input name
    newLayer_next->_inputName = newLayer->_name;
    //insert one node
    newLayer_next->insertPrevLayer(newLayer);
    cout<<"create layer success"<<endl;
}

/*softMaxLayer fission*/
void softmaxFission(LayersBase* splitNode)
{
    //modify config
    configBase* configFiss = FissionFactory::instanceObject()->createConfig(splitNode);
    configBase* curConfig = config::instanceObjtce()->getLayersByName(splitNode->_name);
    configBase* prevConfig = config::instanceObjtce()->getLayersByName(splitNode->_inputName);
    //necessary
    configFiss->_prev.clear();
    configFiss->_next.clear();

    prevConfig->_next.push_back(configFiss);
    configFiss->_prev.push_back(prevConfig);
    config::instanceObjtce()->insertLayerByName(configFiss->_name, configFiss);


    //modify Layers
    LayersBase* layerFiss = FissionFactory::instanceObject()->createLayer(splitNode);
    layerFiss->nextLayer.clear();
    layerFiss->prevLayer.clear();
    //layerFiss
    LayersBase* prevLayer = Layers::instanceObject()->getLayer(splitNode->_inputName);
    //prev
    Layers::instanceObject()->storLayers(prevLayer->_name, layerFiss->_name,layerFiss);
}
