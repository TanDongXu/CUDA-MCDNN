#include"Inception.h"

Inception::Inception(convLayerBase* prevLayer, int sign, const param_tuple& args)
{
	dstData = NULL;
	std::tie(one, three, five, three_reduce, five_reduce, pool_proj,
			inputAmount, inputImageDim, epsilon, lrate, lambda) = args;

	InnerLayers = new Layers[4];

//    first_ShareLayer = new ShareLayer("share1", prevLayer);
//
//    /*the first layer is share layer*/
//	for(int i = 0; i < 4; i++)
//	{
//		sprintf(branch, "branch_%d", i);
//		InnerLayers[i].storLayers(branch, new ShareLayer(branch, prevLayer));
//	}

	Conv_one = new convLayer("one", sign,
			   convLayer::param_tuple(0, 0, 1, 1, 1, one, inputAmount, inputImageDim, epsilon, lrate, lambda));

	Conv_three_reduce = new convLayer("three_reduce", sign,
			            convLayer::param_tuple(0, 0, 1, 1, 1, three_reduce, inputAmount, inputImageDim, epsilon, lrate, lambda));

	Conv_three = new convLayer("three", sign,
			     convLayer::param_tuple(1, 1, 1, 1, 3, three, three_reduce, inputImageDim - 3 + 1, epsilon, lrate, lambda));

	Conv_five_reduce = new convLayer("five_reduce", sign,
			           convLayer::param_tuple(0, 0, 1, 1, 1, five_reduce, inputAmount, inputImageDim, epsilon, lrate, lambda));

	Conv_five = new convLayer("five", sign,
			    convLayer::param_tuple(2, 2, 1, 1, 5, five, five_reduce, inputImageDim - 5 + 1, epsilon, lrate, lambda));

	max_pool = new poolLayer("max_pool", poolLayer::param_tuple("max", 3, 1, 1, 1, 1, inputAmount, inputImageDim));

	Conv_pool_proj = new convLayer("pool_proj",sign,
			         convLayer::param_tuple(0, 0, 1, 1, 1, pool_proj, inputAmount, inputImageDim/3, epsilon,lrate, lambda));

	/*主要用于反向传导*/
	share_Layer = new ShareLayer("share");

	InnerLayers[0].storLayers("one", Conv_one);
	InnerLayers[1].storLayers("three_reduce", Conv_three_reduce);
	InnerLayers[1].storLayers("three", Conv_three);
	InnerLayers[2].storLayers("five_reduce", Conv_five_reduce);
	InnerLayers[2].storLayers("five", Conv_five);
	InnerLayers[3].storLayers("max_pool", max_pool);
	InnerLayers[3].storLayers("pool_proj", Conv_pool_proj);

	/*the last layer is shared layer*/
	for(int i = 0; i < 4; i++)
	{
		//InnerLayers[i].storLayers("share", share_Layer);
	    InnerLayers[i].getLayer(InnerLayers[i].getLayersName(0))->prevLayer = prevLayer;
	    InnerLayers[i].getLayer(InnerLayers[i].getLayersName(InnerLayers[i].getLayersNum() - 1))->nextLayer = share_Layer;
	}

	concat = new Concat(InnerLayers, Concat::param_tuple(one, three, five, pool_proj));

}



void Inception::forwardPropagation(string train_or_test)
{
    layersBase* layer;

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < InnerLayers[i].getLayersNum(); j++)
		{
            layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(j));
            layer->forwardPropagation(train_or_test);

            if(j > 0)
            {
            	layer->Forward_cudaFree();
            }
		}
	}


	dstData = concat->forwardSetup();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < InnerLayers[i].getLayersNum(); j++)
		{
			layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(j));
			if (j == InnerLayers[i].getLayersNum() - 1)
			{
				MemoryMonitor::instanceObject()->freeGpuMemory(layer->dstData);
			}
		}
	}

}





void Inception::backwardPropagation(float*& nextLayerDiffData, float Momentum)
{
	share_Layer->diffData = nextLayerDiffData;
	layersBase* layer;
	/*the first layer no need compute diffData here*/
	for(int i = 0; i < 4; i++)
	{
		for(int j = InnerLayers[i].getLayersNum() - 1; j >= 0; j--)
		{
	        layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(j));

	        if(layer->_name == "three")
	        {
	        	cout<<(share_Layer->diffData == NULL)<<endl;
	        }

	        cout<<layer->_name<<endl;
	        layer->backwardPropagation(Momentum);


	        if(i != 0 && j == 0)
	        {

	            layer->Backward_cudaFree();
	        }
		}
	}

	diffData = concat->backwardSetup();

	for (int i = 0; i < 4; i++)
	{
		layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(0));
		MemoryMonitor::instanceObject()->freeGpuMemory(layer->diffData);
	}
}
