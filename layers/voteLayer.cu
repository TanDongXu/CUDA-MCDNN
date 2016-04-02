#include "voteLayer.h"

VoteLayer::VoteLayer(){
    m_nNumOfTestData = -1;
    m_nClasses = -1;
    m_pHostVote = NULL;
}

VoteLayer* VoteLayer::instance(){
    static VoteLayer tmp;
    return &tmp;
}

void VoteLayer::vote( int nStartPosition, int nBatchSize, float* pDevVote )
{
    int nRemain = nBatchSize;
    if( nBatchSize + nStartPosition * nBatchSize > m_nNumOfTestData ){
        nRemain = m_nNumOfTestData - nStartPosition * nBatchSize;
    }
    else if ( nStartPosition * nBatchSize == m_nNumOfTestData ){
        return;
    }
    //printf("startPosition %d %d %d\n", m_nNumOfTestData, nStartPosition, nRemain);

    if( nRemain <= 0 )
    {
        printf("VoteLayer:vote error, nRemain = 0\n");
        exit(0);
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
        //printf("%d\n", nCurPosition);
        if( nCurPosition >= m_nNumOfTestData * m_nClasses ){
            printf(" nCurPosition >= m_nNumOfTestData\n");
            exit(0);
        }
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
    //printf("%d %d\n",m_nNumOfTestData, m_nClasses);

    for(int i = 0; i < m_nNumOfTestData; i++){
        float fMax = -1;
        int nIndex =  -1;
        for(int j = 0; j < m_nClasses; j++){
            float fValue = m_pHostVote[i * m_nClasses + j];
            //printf("%f\n", fValue);
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
        printf("m_nNumOfTestData %d\n", m_nNumOfTestData);

        m_pHostVote = (float*) MemoryMonitor::instanceObject()->cpuMallocMemory(m_nNumOfTestData * m_nClasses * sizeof(float));
    }
    else
    {
        printf("VoteLayers has be initialized\n");
        exit(0);
    }
}
