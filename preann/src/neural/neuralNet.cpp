#include "neuralNet.h"

NeuralNet::NeuralNet(ImplementationType implementationType)
{
    this->implementationType = implementationType;
}

NeuralNet::~NeuralNet()
{
    CLEAR_PTR_VECTOR(InputLayer, inputs)
    CLEAR_PTR_VECTOR(Layer, layers)
}

Layer* NeuralNet::getLayer(unsigned pos)
{
    return layers[pos];
}

void NeuralNet::calculateOutput()
{
    for (unsigned i = 0; i < inputs.size(); i++) {
        inputs[i]->calculateOutput();
    }
    for (unsigned i = 0; i < layers.size(); i++) {
        layers[i]->calculateOutput();
    }
}

void NeuralNet::addLayer(unsigned size, BufferType destinationType,
        FunctionType functiontype)
{
    layers.push_back(new Layer(size, destinationType, functiontype,
            getImplementationType()));
}

void NeuralNet::addInputLayer(unsigned size, BufferType bufferType)
{
    std::string
            error =
                    "NeuralNet::addInputLayer(unsigned size, BufferType bufferType) not implemented.";
    throw error;
    //TODO quitar esto
    //	inputs.push_back(layers.size());
    //	addLayer(new InputLayer(size, bufferType, getImplementationType()));
}

void NeuralNet::addInputLayer(Interface* interface)
{
    InputLayer* inputLayer = new InputLayer(interface, getImplementationType());
    inputs.push_back(inputLayer);
}

void NeuralNet::updateInput(unsigned inputPos, Interface* input)
{
    if (inputPos > inputs.size()) {
        std::string error = "Cannot get the Input in position " + to_string(
                inputPos) + ": there are just " + to_string(inputs.size())
                + " Inputs.";
        throw error;
    }
    inputs[inputPos]->getInputInterface()->copyFrom(input);
}

unsigned NeuralNet::getNumInputs()
{
    return inputs.size();
}

Interface* NeuralNet::getOutput(unsigned layerPos)
{
    if (layerPos >= layers.size()) {
        std::string error = "Cannot access the output in position "
                + to_string(layerPos) + ": there are just " + to_string(
                layers.size()) + " layers.";
        throw error;
    }
    return layers[layerPos]->getOutputInterface();
}

unsigned NeuralNet::getNumLayers()
{
    return layers.size();
}

void NeuralNet::randomWeighs(float range)
{
    for (unsigned i = 0; i < layers.size(); i++) {
        layers[i]->randomWeighs(range);
    }
}

void NeuralNet::addInputConnection(unsigned sourceInputPos,
        unsigned destinationLayerPos)
{
    if (sourceInputPos >= inputs.size()) {
        std::string error =
                "NeuralNet::addInputConnection : Cannot connect Input in position "
                        + to_string(sourceInputPos) + ": there are just "
                        + to_string(inputs.size()) + " Inputs.";
        throw error;
    }
    if (destinationLayerPos >= layers.size()) {
        std::string error =
                "NeuralNet::addInputConnection : Cannot connect Layer in position "
                        + to_string(destinationLayerPos) + ": there are just "
                        + to_string(layers.size()) + " Layers.";
        throw error;
    }

    layers[destinationLayerPos]->addInput(inputs[sourceInputPos]);
    inputConnectionsGraph.addConnection(sourceInputPos, destinationLayerPos);
}

void NeuralNet::addLayersConnection(unsigned sourceLayerPos,
        unsigned destinationLayerPos)
{
    if (sourceLayerPos >= layers.size() || destinationLayerPos >= layers.size()) {
        std::string error =
                "NeuralNet::addLayersConnection : Cannot connect Layer in position "
                        + to_string(sourceLayerPos)
                        + " with Layer in position " + to_string(
                        destinationLayerPos) + ": there are just " + to_string(
                        layers.size()) + " Layers.";
        throw error;
    }

    layers[destinationLayerPos]->addInput(layers[sourceLayerPos]);
    connectionsGraph.addConnection(sourceLayerPos, destinationLayerPos);
}

void NeuralNet::createFeedForwardNet(unsigned inputSize, BufferType inputType,
        unsigned numLayers, unsigned sizeLayers, BufferType hiddenLayersType,
        FunctionType functiontype)
{
    addInputLayer(inputSize, inputType);
    for (unsigned i = 0; i < numLayers; i++) {
        addLayer(sizeLayers, hiddenLayersType, functiontype);
        addLayersConnection(i, i + 1);
    }
}

ImplementationType NeuralNet::getImplementationType()
{
    if (layers.size() > 0) {
        return layers[0]->getImplementationType();
    } else {
        return implementationType;
    }
}

void NeuralNet::createFullyConnectedNet(unsigned inputSize,
        BufferType inputType, unsigned numLayers, unsigned sizeLayers,
        BufferType hiddenLayersType, FunctionType functiontype)
{
    addInputLayer(inputSize, inputType);
    for (unsigned i = 0; i < numLayers; i++) {
        addLayer(sizeLayers, hiddenLayersType, functiontype);
    }

    for (unsigned src = 0; src <= numLayers; src++) {
        for (unsigned dest = 1; dest <= numLayers; dest++) {
            addLayersConnection(src, dest);
        }
    }
}

void NeuralNet::save(FILE* stream)
{
    unsigned numInputs = inputs.size();
    unsigned numLayers = layers.size();
    fwrite(&numInputs, sizeof(unsigned), 1, stream);
    fwrite(&numLayers, sizeof(unsigned), 1, stream);

    for (unsigned i = 0; i < numInputs; i++) {
        inputs[i]->save(stream);
    }
    for (unsigned i = 0; i < numLayers; i++) {
        layers[i]->save(stream);
    }

    inputConnectionsGraph.save(stream);
    connectionsGraph.save(stream);

    for (unsigned i = 0; i < numLayers; i++) {
        layers[i]->saveWeighs(stream);
    }
}

void NeuralNet::load(FILE* stream)
{
    unsigned numInputs, numLayers;
    fread(&numInputs, sizeof(unsigned), 1, stream);
    fread(&numLayers, sizeof(unsigned), 1, stream);

    for (unsigned i = 0; i < numInputs; i++) {
        inputs.push_back(new InputLayer(stream, getImplementationType()));
    }
    for (unsigned i = 0; i < numLayers; i++) {
        layers.push_back(new Layer(stream, getImplementationType()));
    }

    inputConnectionsGraph.load(stream);
    connectionsGraph.load(stream);

    stablishConnections();

    for (unsigned i = 0; i < layers.size(); i++) {
        layers[i]->loadWeighs(stream);
    }
}

void NeuralNet::stablishConnections()
{
    std::vector<std::pair<unsigned, unsigned> >::iterator it;
    for (it = inputConnectionsGraph.getIterator(); it
            != inputConnectionsGraph.getEnd(); ++it) {
        addInputConnection(it->first, it->second);
    }
    for (it = connectionsGraph.getIterator(); it != connectionsGraph.getEnd(); ++it) {
        addLayersConnection(it->first, it->second);
    }
}
