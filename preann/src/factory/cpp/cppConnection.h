#ifndef CPPCONNECTION_H_
#define CPPCONNECTION_H_

#include "neural/connection.h"
#include "cppBuffer.h"

template<BufferType bufferTypeTempl, class c_typeTempl>
    class CppConnection : public virtual Connection, public CppBuffer<bufferTypeTempl, c_typeTempl>
    {
    protected:

        virtual void _calculateAndAddTo(Buffer* resultsVect)
        {
            float* results = (float*) resultsVect->getDataPointer();
            unsigned inputSize = tInput->getSize();

            switch (tInput->getBufferType()) {
                case BT_BYTE:
                    {
                        std::string error =
                                "CppConnection::_calculateAndAddTo is not implemented for BufferType BYTE as input.";
                        throw error;
                    }
                case BT_FLOAT:
                    {
                        float* inputWeighs = (float*) this->getDataPointer();
                        float* inputPtr = (float*) tInput->getDataPointer();
                        for (unsigned j = 0; j < resultsVect->getSize(); j++) {
                            for (unsigned k = 0; k < inputSize; k++) {
                                results[j] += inputPtr[k] * inputWeighs[(j * inputSize) + k];
                            }
                        }
                    }
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        unsigned char* inputWeighs = (unsigned char*) this->getDataPointer();
                        unsigned* inputPtr = (unsigned*) tInput->getDataPointer();

                        for (unsigned j = 0; j < resultsVect->getSize(); j++) {
                            for (unsigned k = 0; k < inputSize; k++) {
                                unsigned weighPos = (j * inputSize) + k;
                                if (inputPtr[k / BITS_PER_UNSIGNED]
                                        & (0x80000000 >> (k % BITS_PER_UNSIGNED))) {
                                    results[j] += inputWeighs[weighPos] - 128;
                                } else if (tInput->getBufferType() == BT_SIGN) {
                                    results[j] -= inputWeighs[weighPos] - 128;
                                }
                            }
                        }
                    }
                    break;
            }
        }

        virtual void _activation(Buffer* output, FunctionType functionType)
        {
            float* results = (float*) tInput->getDataPointer();
            float* threesholds = (float*) data;

            switch (output->getBufferType()) {
                case BT_BYTE:
                    {
                        std::string error =
                                "CppConnection::_activation is not implemented for an output of BufferType BYTE.";
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
                        unsigned* outputData = (unsigned*) output->getDataPointer();
                        unsigned mask;
                        for (unsigned i = 0; i < tSize; i++) {

                            if (i % BITS_PER_UNSIGNED == 0) {
                                mask = 0x80000000;
                            } else {
                                mask >>= 1;
                            }

                            if (results[i] - threesholds[i] > 0) {
                                outputData[i / BITS_PER_UNSIGNED] |= mask;
                            } else {
                                outputData[i / BITS_PER_UNSIGNED] &= ~mask;
                            }
                        }
                    }
                    break;
            }
        }

        virtual void _crossover(Buffer* other, Interface* bitBuffer)
        {
            switch (bufferTypeTempl) {
                case BT_BIT:
                case BT_SIGN:
                    {
                        std::string error =
                                "CppBuffer::_crossover is not implemented for BufferType BIT nor SIGN.";
                        throw error;
                    }
                default:
                    {
                        //TODO Z decidir cual mola mas
                        c_typeTempl* otherWeighs = other->getDataPointer2<c_typeTempl>();
                        c_typeTempl* thisWeighs = (c_typeTempl*) this->getDataPointer();
                        c_typeTempl auxWeigh;

                        for (unsigned i = 0; i < tSize; i++) {
                            if (bitBuffer->getElement(i)) {
                                auxWeigh = thisWeighs[i];
                                thisWeighs[i] = otherWeighs[i];
                                otherWeighs[i] = auxWeigh;
                            }
                        }
                    }
                    break;
            }
        }

        virtual void _mutateWeigh(unsigned pos, float mutation)
        {
            switch (bufferTypeTempl) {
                case BT_BYTE:
                    {
                        c_typeTempl* weigh = &(((c_typeTempl*) data)[pos]);
                        int result = (int) mutation + *weigh;
                        if (result <= 0) {
                            *weigh = 0;
                        } else if (result >= 255) {
                            *weigh = 255;
                        } else {
                            *weigh = result;
                        }
                    }
                    break;
                case BT_FLOAT:
                    ((c_typeTempl*) data)[pos] += mutation;
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        unsigned mask = 0x80000000 >> (pos % BITS_PER_UNSIGNED);
                        ((unsigned*) data)[pos / BITS_PER_UNSIGNED] ^= mask;
                    }
                    break;
            }
        }

        virtual void _resetWeigh(unsigned pos)
        {
            switch (bufferTypeTempl) {
                case BT_BYTE:
                    {
                        ((c_typeTempl*) data)[pos] = 128;
                    }
                    break;
                case BT_FLOAT:
                    ((c_typeTempl*) data)[pos] = 0;
                    break;
                case BT_BIT:
                case BT_SIGN:
                    {
                        unsigned mask = 0x80000000 >> (pos % BITS_PER_UNSIGNED);
                        ((unsigned*) data)[pos / BITS_PER_UNSIGNED] &= ~mask;
                    }
                    break;
            }
        }

    public:
        virtual ~CppConnection()
        {
        }
        ;

        CppConnection(Buffer* input, unsigned outputSize)
                : CppBuffer<bufferTypeTempl, c_typeTempl>(input->getSize() * outputSize)
        {
            tInput = input;
        }
    };

#endif /* CPPCONNECTION_H_ */
