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


	void push_back(cuMatrix<T>* m)
	{
		m_vec.push_back(m);
	}

	size_t size()
	{
		return m_vec.size();
	}


//	/*复制每个cuMatrix在GPU上的地址拷贝都GPU*/
//	void toGpu()
//	{
//		 /*m_hostPoint 存储的是指针*/
//		 m_hostPoint = (T **)MemoryMonitor::instanceObject()->cpuMallocMemory(m_vec.size() * sizeof(T*));
//
//		if(!m_hostPoint)
//		{
//			printf("cuMaetrixVector: allocate m_hostPoint Failed\n");
//			exit(0);
//		}
//
//	    MemoryMonitor::instanceObject()->gpuMallocMemory((void**)&m_devPoint,sizeof(T*) * m_vec.size());
//
//		/*m_hostPoint 存储的是数据在GPU上的指针
//		 * 将每个cuMatrix在GPU上的地址赋给m_hostPoint
//		 * 注意此时必须先把每个cuMatrix存储到GPU上
//		 * */
//	     for(int p=0; p<m_vec.size(); p++)
//		 {
//			 m_hostPoint[p] = m_vec[p]->getDeviceData();
//
//		}
//
//		checkCudaErrors(cudaMemcpy(m_devPoint, m_hostPoint, sizeof(T*) * m_vec.size(), cudaMemcpyHostToDevice));
//	}


    /*m_hostPoint 存储的是数据在GPU上的指针*/
	T** m_hostPoint;
	/*m_devPoint 也是存储的是数据在GPU上的指针*/
	T** m_devPoint;
	vector<cuMatrix<T>* > m_vec;

};


#endif /* CUMATRIXVECTOR_H_ */
