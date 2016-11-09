#include"LayersBase.h"
#include"../config/config.h"

/*
 * Set branch index
 * */
void LayersBase::setCurBranchIndex(int nIndex)
{
	m_nCurBranchIndex = nIndex;
}

/*
 * Adjust learning rate
 * */
void LayersBase::adjust_learnRate(int index, double lr_gamma, double lr_power)
{
    lrate = static_cast<float>(lrate * pow((1.0 + lr_gamma * index), (-lr_power)));
}

/*Insert previous layer*/
void LayersBase::insertPrevLayer(LayersBase* layer)
{
    prevLayer.push_back(layer);
}

/*
 * Insert next layer
 * */
void LayersBase::insertNextlayer(LayersBase* layer)
{
    nextLayer.push_back(layer);
}

/*
 * Learning rate reduce
 * */
void LayersBase::rateReduce()
{
    if( lrate < 1 )
    {
    //	if( m_fReduceRate <= 1.0f)
    //    lrate /= 2.0f;
    //    else if( m_fReduceRate <= 2.0f)
    //    	lrate /= 2.5f;
    //    else if( m_fReduceRate <= 5.0f)
    //        lrate /= 3.0f;
        lrate /= m_fReduceRate;
    }
}

void LayersBase::setRateReduce( float fReduce)
{
    m_fReduceRate = fReduce;
}

float LayersBase::getRateReduce()
{
    return m_fReduceRate;
}

/*
 * Get one layer from map
 * */
LayersBase* Layers::getLayer(string name)
{
    if(_layersMaps.find(name) != _layersMaps.end())
    {
        return _layersMaps[name];
    }else
    {
        printf("layer: get layer %s is not exist\n",name.c_str());
        exit(0);
        return NULL;
    }
}

/*
 *The layer is in Map or no
 * */
bool Layers::hasLayer(string name)
{
    if(_layersMaps.find(name) != _layersMaps.end())
    {
        return true;
    }else
    {
        return false;
    }
}

/*
 * Story the layername into vector
 * */
void Layers::storLayersName(string name)
{
    _layersName.push_back(name);
}

/*
 * Story the layers into map
 * */
void Layers::storLayers(string prev_name, string name, LayersBase* layer)
{
    if(_layersMaps.find(name) == _layersMaps.end())
    {
        _layersMaps[name] = layer;
        storLayersName(name);

        /*create a linked list*/
        if(string("NULL") == prev_name)
        {
            _layersMaps[name]->prevLayer.clear();
            _layersMaps[name]->_inputName = " ";
        }else
        {
            _layersMaps[name]->_inputName = prev_name;
            //cout<<"prevName: "<<prev_name<<" name: "<<name<<endl;
            _layersMaps[prev_name]->insertNextlayer( _layersMaps[name] );
            _layersMaps[name]->insertPrevLayer(_layersMaps[prev_name]);
        }

    }else
    {
        printf("layers: the layer %s have already in layersMap\n",name.c_str());
        exit(0);
    }
}

/*
 * overload
 * Linear storage layer
 */
void Layers::storLayers(string name, LayersBase* layer)
{
    if(_layersMaps.find(name) == _layersMaps.end())
    {
        _layersMaps[name] = layer;
        storLayersName(name);

        /*create a linked list*/
        if(_layersMaps.size() == 1)
        {
            _layersMaps[name]->prevLayer.clear();
            _layersMaps[name]->_inputName = " ";

        }else
        {
            _layersMaps[name]->_inputName = _layersMaps[_layersName[_layersName.size() - 2]]->_name;
            _layersMaps[_layersName[_layersName.size() -2 ]]->insertNextlayer( _layersMaps[name]  );
            _layersMaps[name]->insertPrevLayer(_layersMaps[_layersName[_layersName.size() - 2]]);
        }
    }else
    {
        printf("layers: the layer %s have already in layersMap\n",name.c_str());
        exit(0);

    }
}

/*Get the name of layer*/
string Layers::getLayersName(int index)
{
    if(index >= _layersName.size())
    {
        printf("layers: the index %d has already out of layersName size\n", index);
        exit(0);
    }else
    {
        return _layersName[index];
    }
}
