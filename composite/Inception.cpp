#include"Inception.h"


/*Inception constructor*/
Inception::Inception(convLayerBase* prevLayer, 
                     int sign, 
                     float* rate, 
                     const param_tuple& args)
{
	std::tie(one, three, five, three_reduce, five_reduce, pool_proj,
		     inputAmount, inputImageDim, epsilon, lambda) = args;

	dstData = NULL;
	lrate = rate;
	InnerLayers = new Layers[4];

	Conv_one = new convLayer("one", sign,
		       convLayer::param_tuple(0, 0, 1, 1, 1, 
                                     one,
					         inputAmount, 
                             inputImageDim, 
                             epsilon, 
                             *lrate, 
                             lambda));

	Conv_three_reduce = new convLayer("three_reduce", sign,
			            convLayer::param_tuple(0, 0, 1, 1, 1,
                                      three_reduce,
			            		       inputAmount, 
                                      inputImageDim, 
                                      epsilon, 
                                      *lrate, 
                                      lambda));

	Conv_three = new convLayer("three", sign,
			     convLayer::param_tuple(1, 1, 1, 1, 3,
                               three,
			    		       three_reduce, 
                               inputImageDim, 
                               epsilon, 
                               *lrate, 
                               lambda));

	Conv_five_reduce = new convLayer("five_reduce", sign,
			           convLayer::param_tuple(0, 0, 1, 1, 1, 
                                     five_reduce,
			        		         inputAmount, 
                                     inputImageDim, 
                                     epsilon, 
                                     *lrate, 
                                     lambda));

	Conv_five = new convLayer("five", sign,
			    convLayer::param_tuple(2, 2, 1, 1, 5, 
                              five,
			    		      five_reduce, 
                              inputImageDim,
                              epsilon, 
                              *lrate, 
                              lambda));

	max_pool = new poolLayer("max_pool", 
               poolLayer::param_tuple("max", 3, 1, 1, 1, 1, 
                                      inputImageDim, 
                                      inputAmount));

	Conv_pool_proj = new convLayer("pool_proj",sign,
			         convLayer::param_tuple(0, 0, 1, 1, 1,
                                   pool_proj,
			        		       inputAmount, 
                                   inputImageDim, 
                                   epsilon, 
                                   *lrate, 
                                   lambda));

	/*mainly use in backpropagation*/
	share_Layer = new ShareLayer("share");

   /*four branch*/
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
	    InnerLayers[i].getLayer(InnerLayers[i].getLayersName(0))->prevLayer = prevLayer;
	    InnerLayers[i].getLayer(InnerLayers[i].getLayersName(InnerLayers[i].getLayersNum() - 1))->nextLayer = share_Layer;
	}


	concat = new Concat(InnerLayers, Concat::param_tuple(one, three, five, pool_proj));

}


/*Inception forwardPropagation*/
void Inception::forwardPropagation(string train_or_test)
{
    layersBase* layer;

	for(int i = 0; i < 4; i++)
	{
		for(int j = 0; j < InnerLayers[i].getLayersNum(); j++)
		{
            layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(j));
            layer->lrate = *lrate;
            layer->forwardPropagation(train_or_test);
            if(j > 0 && train_or_test == "test")
            {
            	layer->Forward_cudaFree();
            }
		}
	}

	/*get the inception result data*/
	dstData = NULL;
	dstData = concat->forwardSetup();

	if (train_or_test == "test")
	{
		for (int i = 0; i < 4; i++)
		{
			layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(InnerLayers[i].getLayersNum() - 1));
			MemoryMonitor::instanceObject()->freeGpuMemory(layer->dstData);
		}
	}
}


/*inception backwardPropagation*/
void Inception::backwardPropagation(float*& nextLayerDiffData, float Momentum)
{
	layersBase* layer;
	for(int i = 0; i < 4; i++)
	{
		concat->split_DiffData(i, nextLayerDiffData);

		for(int j = InnerLayers[i].getLayersNum() - 1; j >= 0; j--)
		{
	        layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(j));
	        layer->backwardPropagation(Momentum);
	        layer->Backward_cudaFree();
		}
	}

	/*get inception diff*/
	diffData = NULL;
	diffData = concat->backwardSetup();

	for (int i = 0; i < 4; i++)
	{
		/*free first layer diffData*/
		layer = InnerLayers[i].getLayer(InnerLayers[i].getLayersName(0));
		MemoryMonitor::instanceObject()->freeGpuMemory(layer->diffData);
	}
}
