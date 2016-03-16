#include"saveNetWork.h"

/*save network parameter*/
void saveNetWork()
{
	char fileName[50];
	int layerNum = Layers::instanceObject()->getLayersNum();
	int imageSize = config::instanceObjtce()->get_imageSize();
	int normalized_width = config::instanceObjtce()->get_normalizedWidth();

	sprintf(fileName,"models/net_normalized_%d_%d.txt", imageSize, normalized_width);
	FILE *file = fopen(fileName, "w");

	if (file != NULL)
	{
		configBase *config = (configBase*) config::instanceObjtce()->getFirstLayers();
		for (int i = 0; i < layerNum; i++) {
			layersBase *layer = (layersBase*) Layers::instanceObject()->getLayer(config->_name);
			layer->saveWeight(file);
			config = config->_next;
		}

	}else{
		cout<<"savaNetWork:: Open Failed"<<endl;
		exit(0);
	}
	fclose(file);
	file = NULL;
}
