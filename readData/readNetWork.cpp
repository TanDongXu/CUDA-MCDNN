#include"readNetWork.h"


/*read network parameter*/
void readNetWork()
{

	char fileName[50];
	int layerNum = Layers::instanceObject()->getLayersNum();
	int imageSize = config::instanceObjtce()->get_imageSize();
	int normalized_width = config::instanceObjtce()->get_normalizedWidth();

	sprintf(fileName, "models/net_normalized_%d_%d.txt", imageSize, normalized_width);
	FILE *file = fopen(fileName, "r");

	if (file != NULL) {
		configBase *config = (configBase*) config::instanceObjtce()->getFirstLayers();
		for (int i = 0; i < layerNum; i++) {
			layersBase *layer = (layersBase*) Layers::instanceObject()->getLayer(config->_name);
			layer->readWeight(file);
			config = config->_next;
		}

	} else {
		cout << "readNetWork:: Open Failed" << endl;
		exit(0);
	}
	fclose(file);
	file = NULL;

}
