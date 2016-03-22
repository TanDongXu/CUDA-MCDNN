#ifndef VOTE_LAYER_H
#define VOTE_LAYER_H

#include "layersBase.h"
#include "../common/cuMatrix.h"
#include "../config/config.h"
#include <string>

class VoteLayer
{
public:
    static VoteLayer* instance();
    VoteLayer();
    ~VoteLayer();
    void vote( int n_start_position, int n_batch_size, float* p_host_vote );
    float result();
    void init(int nNumOfTestData, int nClasses, cuMatrix<int>* pLabels);
    void clear();
private:
    int m_nNumOfTestData;
    int m_nClasses;
    float* m_pHostVote;// m_nNumOfTestData * m_nClasses
    int* m_pLabels; //labels
};

#endif 

