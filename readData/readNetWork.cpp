#include"readNetWork.h"
#include <queue>
#include <set>

/*read network parameter*/
void readNetWork()
{
	char fileName[50];
	int layerNum = Layers::instanceObject()->getLayersNum();
	int imageSize = config::instanceObjtce()->get_imageSize();
	int normalized_width = config::instanceObjtce()->get_normalizedWidth();

	sprintf(fileName, "models/net_normalized_%d_%d.txt", imageSize, normalized_width);
	FILE *file = fopen(fileName, "r");
    queue<configBase*> que;
    set<configBase*> hash;
    
	if (file != NULL) {
		configBase *config = (configBase*) config::instanceObjtce()->getFirstLayers();
        que.push(config);
        hash.insert(config);
        while( !que.empty() ){
            config = que.front();
            que.pop();
			layersBase *layer = (layersBase*) Layers::instanceObject()->getLayer(config->_name);
			layer->readWeight(file);
            for(int i = 0; i < config->_next.size(); i++){
                configBase* tmp = config->_next[i];
                if( hash.find(tmp) != hash.end()){
                    hash.insert(tmp);
                    que.push(tmp);
                }
            }
		}
	} else {
		cout << "readNetWork:: Open Failed" << endl;
		exit(0);
	}
	fclose(file);
	file = NULL;

}
