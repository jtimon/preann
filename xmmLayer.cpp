#include "xmmLayer.h"

XmmLayer::XmmLayer()
{
}

XmmLayer::~XmmLayer()
{
	if (inputs) {
		for (unsigned i=0; i < numberInputs; i++){
			delete(connections[i]);
		}
		mi_free(inputs);
		mi_free(connections);
		inputs = NULL;
		connections = NULL;
	}
	if (thresholds) {
		delete(thresholds);
		thresholds = NULL;
	}
	if (output) {
		delete (output);
		output = NULL;
	}
}

