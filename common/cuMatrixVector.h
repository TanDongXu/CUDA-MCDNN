/*
 * cuMatrixVector.h
 *
 *  Created on: Nov 19, 2015
 *      Author: tdx
 */

#ifndef CUMATRIXVECTOR_H_
#define CUMATRIXVECTOR_H_

#include"MemoryMonitor.h"
#include"cuMatrix.h"
#include<vector>
#include<iostream>
using namespace std;

template <typename T>
class cuMatrixVector
{
public:
	cuMatrixVector():m_hostPoint(0),m_devPoint(0){}

	~cuMatrixVector()
	{
		MemoryMonitor::instanceObject()->freeCpuMemory(m_hostPoint);
		MemoryMonitor::instanceObject()->freeGpuMemory(m_devPoint);
		m_vec.clear();

	}

	/*overload operator []*/
	cuMatrix<T>* operator[](size_t index)
	{
		if(index > m_vec.size())
		{
			cout<<"cuMatrixVector:operator[] error "<<endl;
			exit(0);
		}

		return m_vec[index];
	}


    //push_back
	void push_back(cuMatrix<T>* m)
	{
		m_vec.push_back(m);
	}

    //get size
	size_t size()
	{
		return m_vec.size();
	}




public:
	T** m_hostPoint;
	T** m_devPoint;
	vector<cuMatrix<T>* > m_vec;

};



#endif /* CUMATRIXVECTOR_H_ */
