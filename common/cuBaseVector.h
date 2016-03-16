/*
 * cuBaseVector.h
 *
 *  Created on: Mar 13, 2016
 *      Author: tdx
 */

#ifndef CUBASEVECTOR_H_
#define CUBASEVECTOR_H_

#include"MemoryMonitor.h"
#include<vector>
#include<string>

using namespace std;

template <typename T>

class cuBaseVector
{
public:
	cuBaseVector():hostPoint(0),devPoint(0){}

	~cuBaseVector()
	{

	}

	void vector_clear()
	{
		if (NULL != hostPoint)
		{
			MemoryMonitor::instanceObject()->freeCpuMemory(hostPoint);
			hostPoint = NULL;
		}
		if (NULL != devPoint)
		{
			MemoryMonitor::instanceObject()->freeGpuMemory(devPoint);
			devPoint = NULL;
		}

		vector<T*> a;
		m_vec.swap(a);
	}

	/*overload operator []*/
	T* operator[](size_t index)
	{
		if (index > m_vec.size())
		{
			cout << "cuBaseVector:operator[] error " << endl;
			exit(0);
		}

		return m_vec[index];
	}

	void push_back(T* m) {
		m_vec.push_back(m);
	}

	size_t size()
	{
		return m_vec.size();
	}

	/*复制每个T在GPU上的地址拷贝都GPU*/
	void toGpu()
	{
		/*hostData 存储的是指针*/
		hostPoint = (T**) MemoryMonitor::instanceObject()->cpuMallocMemory(m_vec.size() * sizeof(T*));

		if (!hostPoint) {
			printf("cuBaseVector: allocate m_hostPoint Failed\n");
			exit(0);
		}

		devPoint = NULL;
		MemoryMonitor::instanceObject()->gpuMallocMemory((void**) &devPoint, sizeof(T*) * m_vec.size());

		/*hostData 存储的是数据在GPU上的指针
		 * 将T在GPU上的地址赋给hostData
		 * 注意此时必须先把每个地址存储到GPU上
		 * */
		for (int p = 0; p < m_vec.size(); p++) {
			hostPoint[p] = m_vec[p];

		}

		checkCudaErrors(cudaMemcpy(devPoint, hostPoint, sizeof(T*) * m_vec.size(), cudaMemcpyHostToDevice));
	}



public:
	T** hostPoint;
	T** devPoint;

private:

	vector<T*> m_vec;

};


#endif /* CUBASEVECTOR_H_ */
