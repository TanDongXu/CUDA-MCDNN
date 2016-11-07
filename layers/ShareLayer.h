/*
* ShareLayer.h
*
*  Created on: Mar 7, 2016
*      Author: tdx
*/

#ifndef SHARELAYER_H_
#define SHARELAYER_H_

#include<tuple>
#include<string>
#include"LayersBase.h"

/*
 * Class SharedLayer
 * */
class ShareLayer : public LayersBase
{
    public:
    ShareLayer(string name);
    ShareLayer(string name, LayersBase* layer);
    void forwardPropagation(string train_or_test){};
    void backwardPropagation(float Momemtum){};
    void readWeight(FILE* file){};
    void saveWeight(FILE* file){};
    int getOutputSize();

    private:
    int outputSize;
};


#endif /* SHARELAYER_H_ */
