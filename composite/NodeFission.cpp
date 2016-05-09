#include"NodeFission.h"
#include"../config/config.h"
#include"../layers/layersBase.h"

//node fission
void NodeFission(layersBase* splitNode, layersBase* nextNode) {
    vector<layersBase*>::iterator l_iter;
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


    //modify Layers
    layersBase* layerFiss = FissionFactory::instanceObject()->createLayer(splitNode);
    layerFiss->nextLayer.clear();
    layerFiss->prevLayer.clear();
    //layerFiss
    layersBase* prevLayer = Layers::instanceObject()->getLayer(splitNode->_inputName);
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
for (l_iter = nextNode->prevLayer.begin(); l_iter != nextNode->prevLayer.end(); l_iter++) {
    if ((*l_iter)->_name == (splitNode->_name)) {
        nextNode->prevLayer.erase(l_iter);
        break;
    }
}
//insert one node
nextNode->insertPrevLayer(layerFiss);

}

/*softMaxLayer fission*/
void softmaxFission(layersBase* splitNode)
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
    layersBase* layerFiss = FissionFactory::instanceObject()->createLayer(splitNode);
    layerFiss->nextLayer.clear();
    layerFiss->prevLayer.clear();
    //layerFiss
    layersBase* prevLayer = Layers::instanceObject()->getLayer(splitNode->_inputName);
    //prev
    Layers::instanceObject()->storLayers(prevLayer->_name, layerFiss->_name,layerFiss);
}
