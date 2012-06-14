#ifndef XMMCONNECTION_H_
#define XMMCONNECTION_H_

#include "neural/connection.h"
#include "xmmBuffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class XmmConnection : virtual public Connection, public XmmBuffer<bufferTypeTempl, c_typeTempl>
    {
    protected:
        virtual void _copyFrom(Interface* interface)
        {
            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned offset = 0;
            unsigned inputSize = tInput->getSize();
            unsigned outputSize = tSize / inputSize;
            unsigned elem = 0;

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {
                            ((c_typeTempl*) (data) + offset)[i] = interface->getElement(elem++);
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_FLOAT:
                    offsetPerInput = offsetPerInput / sizeof(c_typeTempl);
                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {
                            ((float*) (data) + offset)[i] = interface->getElement(elem++);
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        std::string error =
                                "XmmConnection::_copyFrom is not implemented for BufferType BIT nor SIGN.";
                        throw error;
                    }
            }
        }

        virtual void _copyTo(Interface* interface)
        {
            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned offset = 0;
            unsigned inputSize = tInput->getSize();
            unsigned outputSize = tSize / inputSize;
            unsigned elem = 0;

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {
                            interface->setElement(elem++, ((c_typeTempl*) (data) + offset)[i]);
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_FLOAT:
                    offsetPerInput = offsetPerInput / sizeof(c_typeTempl);
                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {
                            interface->setElement(elem++, ((c_typeTempl*) (data) + offset)[i]);
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        std::string error =
                                "XmmConnection::_copyTo is not implemented for BufferType BIT nor SIGN.";
                        throw error;
                    }
            }
        }

        virtual void _calculateAndAddTo(Buffer* resultsVect)
        {
            void* inputWeighs = this->getDataPointer();
            float* results = (float*) resultsVect->getDataPointer();
            void* inputPtr = tInput->getDataPointer();

            unsigned numLoops;
            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned weighPos = 0;

            switch (tInput->getBufferType()) {
                case BT_FLOAT:
                    offsetPerInput = offsetPerInput / sizeof(float);
                    numLoops = ((tInput->getSize() - 1) / FLOATS_PER_BLOCK) + 1;
                    for (unsigned j = 0; j < resultsVect->getSize(); j++) {
                        float auxResult;
                        XMMreal(inputPtr, numLoops, (((float*) inputWeighs) + weighPos), auxResult);
                        results[j] += auxResult;
                        weighPos += offsetPerInput;
                    }
                    break;
                case BT_BIT:
                    numLoops = ((tInput->getSize() - 1) / BYTES_PER_BLOCK) + 1;
                    for (unsigned j = 0; j < resultsVect->getSize(); j++) {
                        results[j] += XMMbinario(inputPtr, numLoops,
                                                 (((unsigned char*) inputWeighs) + weighPos));
                        weighPos += offsetPerInput;
                    }
                    break;
                case BT_SIGN:
                    numLoops = ((tInput->getSize() - 1) / BYTES_PER_BLOCK) + 1;
                    for (unsigned j = 0; j < resultsVect->getSize(); j++) {
                        results[j] += XMMbipolar(inputPtr, numLoops,
                                                 (((unsigned char*) inputWeighs) + weighPos));
                        weighPos += offsetPerInput;
                    }
                    break;
                case BT_BYTE:
                    std::string error =
                            "CppBuffer::inputCalculation is not implemented for BufferType BYTE as input.";
                    throw error;
            }
        }

        virtual void _activation(Buffer* output, FunctionType functionType)
        {
            float* results = (float*) tInput->getDataPointer();
            float* threesholds = (float*) data;

            switch (output->getBufferType()) {
                case BT_BYTE:
                    {
                        std::string error = "XmmBuffer::activation is not implemented for BufferType BYTE.";
                        throw error;
                    }
                    break;
                case BT_FLOAT:
                    {
                        float* outputData = (float*) output->getDataPointer();
                        for (unsigned i = 0; i < tSize; i++) {
                            outputData[i] = Function<c_typeTempl>(results[i] - threesholds[i], functionType);
                        }
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        unsigned char* outputData = (unsigned char*) output->getDataPointer();

                        unsigned blockOffset = 0;
                        unsigned bytePos = 0;
                        unsigned char bufferMask = 128;

                        for (unsigned i = 0; i < tSize; i++) {

                            if (results[i] - threesholds[i] > 0) {
                                outputData[blockOffset + bytePos] |= bufferMask;
                            } else {
                                outputData[blockOffset + bytePos] &= ~bufferMask;
                            }

                            if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
                                bytePos = 0;
                                if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
                                    blockOffset += BYTES_PER_BLOCK;
                                    bufferMask = 128;
                                } else {
                                    bufferMask >>= 1;
                                }
                            } else {
                                ++bytePos;
                            }
                        }
                    }
                    break;
            }
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            void* otherWeighs = other->getDataPointer();

            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned offset = 0;
            unsigned inputSize = tInput->getSize();
            unsigned outputSize = tSize / inputSize;
            unsigned elem = 0;

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    unsigned char auxChar;

                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {

                            if (bitBuffer->getElement(elem++)) {
                                auxChar = ((unsigned char*) (data) + offset)[i];
                                ((unsigned char*) (data) + offset)[i] = ((unsigned char*) (otherWeighs)
                                        + offset)[i];
                                ((unsigned char*) (otherWeighs) + offset)[i] = auxChar;
                            }
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_FLOAT:
                    float auxFloat;
                    offsetPerInput = offsetPerInput / sizeof(float);

                    for (unsigned j = 0; j < outputSize; j++) {
                        for (unsigned i = 0; i < inputSize; i++) {

                            if (bitBuffer->getElement(elem++)) {
                                auxFloat = ((float*) (data) + offset)[i];
                                ((float*) (data) + offset)[i] = ((float*) (otherWeighs) + offset)[i];
                                ((float*) (otherWeighs) + offset)[i] = auxFloat;
                            }
                        }
                        offset += offsetPerInput;
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    std::string error =
                            "XmmConnection::_crossover is not implemented for BufferType BIT nor SIGN.";
                    throw error;
            }
        }

        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned outputPos = pos / tInput->getSize();
            unsigned inputPos = pos % tInput->getSize();
            unsigned elem = (outputPos * offsetPerInput) + inputPos;

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    {
                        c_typeTempl *weigh = &(((c_typeTempl*) (data))[elem]);
                        int result = (int) (mutation) + *weigh;
                        if (result <= 0) {
                            *weigh = 0;
                        } else {
                            if (result >= 255) {
                                *weigh = 255;
                            } else {
                                *weigh = result;
                            }
                        }

                    }
                    break;
                case BT_FLOAT:
                    offsetPerInput = offsetPerInput / sizeof(c_typeTempl);
                    elem = (outputPos * offsetPerInput) + inputPos;
                    ((c_typeTempl*) (data))[elem] += mutation;
                    break;
                case BT_BIT:
                case BT_SIGN:
                    std::string error =
                            "XmmConnection::mutate is not implemented for BufferType BIT nor SIGN.";
                    throw error;
            }
        }

        virtual void _resetWeigh(unsigned pos)
        {
            unsigned offsetPerInput = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(tInput->getSize());
            unsigned outputPos = pos / tInput->getSize();
            unsigned inputPos = pos % tInput->getSize();
            unsigned elem = (outputPos * offsetPerInput) + inputPos;

            switch (bufferTypeTempl) {
                case BT_BYTE:
                    ((c_typeTempl*) data)[elem] = 128;
                    break;
                case BT_FLOAT:
                    {
                        offsetPerInput = offsetPerInput / sizeof(c_typeTempl);
                        elem = (outputPos * offsetPerInput) + inputPos;
                        ((c_typeTempl*) (data))[elem] = 0;
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    std::string error =
                            "XmmConnection::resetConnection is not implemented for BufferType BIT nor SIGN.";
                    throw error;
            }
        }

    public:
        XmmConnection(Buffer* input, unsigned outputSize)
        {
            tInput = input;
            this->tSize = input->getSize() * outputSize;

            unsigned byteSize = XmmBuffer<bufferTypeTempl, c_typeTempl>::getByteSize(input->getSize());
            byteSize *= outputSize;
            data = MemoryManagement::malloc(byteSize);

            switch (bufferTypeTempl) {

                case BT_BYTE:
                    SetValueToAnArray<unsigned char>(data, byteSize, 128);
                    break;
                case BT_FLOAT:
                    SetValueToAnArray<float>(data, byteSize / sizeof(float), 0);
                    break;
                case BT_BIT:
                case BT_SIGN:
                    SetValueToAnArray<unsigned>(data, byteSize, 0);
                    break;
            }
        }

        virtual ~XmmConnection()
        {
        }
    };

#endif /* XMMCONNECTION_H_ */
