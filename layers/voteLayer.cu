#include "voteLayer.h"

VoteLayer::VoteLayer(){
    m_nNumOfTestData = -1;
    m_nClasses = -1;
    m_pHostVote = NULL;
}

VoteLayer* VoteLayer::instance(){
    static VoteLayer* tmp = new VoteLayer();
    return tmp;
}

void VoteLayer::vote( int nStartPosition, int nBatchSize, float* pDevVote )
{
    int nRemain = nBatchSize;
    if( nBatchSize + nStartPosition * nBatchSize > m_nNumOfTestData )
    {
        nRemain = m_nNumOfTestData - nStartPosition * nBatchSize;
    } 
    cuMatrix<float>tmp(pDevVote, nRemain, m_nClasses, 1, true);
    tmp.toCpu();

    if (m_pHostVote == NULL) 
    {
        printf("VoteLayer has not be initialized\n");
        exit(0);
    }

    for(int i = 0; i < nRemain; i++){
        int nCurPosition = (nStartPosition * nBatchSize + i) * m_nClasses;
        for(int j = 0; j < m_nClasses; j++){
            m_pHostVote[nCurPosition + j] += tmp.getHostData()[i * m_nClasses + j];
        }
    }
}

void VoteLayer::clear()
{
    memset(m_pHostVote, 0, m_nClasses * m_nNumOfTestData * sizeof(float));
}

float VoteLayer::result()
{
    if ( m_pHostVote == NULL )
    {
        printf("VoteLayer has not be initialized\n");
        exit(0);
    }

    float fResult = 0;

    for(int i = 0; i < m_nNumOfTestData; i++){
        float fMax = -1;
        int nIndex =  -1;
        for(int j = 0; j < m_nClasses; j++){
            float fValue = m_pHostVote[i * m_nClasses + j];
            if( fValue > fMax ){
                fMax = fValue;
                nIndex = j;
            }
        }

        if( nIndex == m_pLabels[i] )
        {
            fResult += 1.0;
        }
    }
    return fResult / m_nNumOfTestData;
}

VoteLayer::~VoteLayer()
{
    MemoryMonitor::instanceObject()->freeCpuMemory( m_pHostVote );
}

void VoteLayer::init(int nNumOfTestData, int nClasses, cuMatrix<int>* pLabels) 
{
    if ( m_pHostVote == NULL )
    {
        m_pLabels = pLabels->getHostData();
        m_nClasses = nClasses;
        m_nNumOfTestData = nNumOfTestData;

        m_pHostVote = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(m_nNumOfTestData * m_nClasses * sizeof(float));
        printf("nClasses %d nNumOfTestData %d\n", nClasses, nNumOfTestData);
    }
    else
    {
        printf("VoteLayers has be initialized\n");
        exit(0);
    }
}
