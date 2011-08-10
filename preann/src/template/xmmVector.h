/*
 * xmmVector.h
 *
 *  Created on: Nov 17, 2009
 *      Author: timon
 */

#ifndef XMMVECTOR_H_
#define XMMVECTOR_H_

#include "vector.h"
#include "sse2_code.h"

template<VectorType vectorTypeTempl, class c_typeTempl>
	class XmmVector : virtual public Vector {
	protected:
		static unsigned getByteSize(unsigned size, VectorType vectorType)
		{
			unsigned numBlocks;
			switch (vectorType)
			{
			case BYTE:
				numBlocks = ((size - 1) / BYTES_PER_BLOCK) + 1;
				break;
			case FLOAT:
				numBlocks = ((size - 1) / FLOATS_PER_BLOCK) + 1;
				break;
			case BIT:
			case SIGN:
				numBlocks = ((size - 1) / BITS_PER_BLOCK) + 1;
				break;
			}
			return numBlocks * BYTES_PER_BLOCK;
		}

		void bitCopyFrom(unsigned char *vectorData, Interface *interface)
		{
			unsigned blockOffset = 0;
			unsigned bytePos = 0;
			unsigned char vectorMask = 128;
			for (unsigned i = 0; i < tSize; i++) {

				if (interface->getElement(i) > 0) {
					vectorData[blockOffset + bytePos] |= vectorMask;
				}
				else {
					vectorData[blockOffset + bytePos] &= ~vectorMask;
				}

				if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
					bytePos = 0;
					if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
						blockOffset += BYTES_PER_BLOCK;
						vectorMask = 128;
					}
					else {
						vectorMask >>= 1;
					}
				}
				else {
					++bytePos;
				}
			}
		}

		void bitCopyTo(unsigned char *vectorData, Interface *interface)
		{
			unsigned blockOffset = 0;
			unsigned bytePos = 0;
			unsigned char vectorMask = 128;
			for (unsigned i = 0; i < tSize; i++) {

				if (vectorData[blockOffset + bytePos] & vectorMask) {
					interface->setElement(i, 1);
				}
				else {
					interface->setElement(i, 0);
				}

				if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
					bytePos = 0;
					if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
						blockOffset += BYTES_PER_BLOCK;
						vectorMask = 128;
					}
					else {
						vectorMask >>= 1;
					}
				}
				else {
					++bytePos;
				}
			}
		}

		virtual void copyFromImpl(Interface* interface)
		{
			switch (vectorTypeTempl)
			{
			default:
				memcpy(data, interface->getDataPointer(),
						interface->getByteSize());
				break;
			case BIT:
			case SIGN:
				unsigned char* vectorData = (unsigned char*)(data);
				bitCopyFrom(vectorData, interface);
			}
		}

		virtual void copyToImpl(Interface* interface)
		{
			switch (vectorTypeTempl)
			{
			default:
				memcpy(interface->getDataPointer(), data,
						interface->getByteSize());
				break;
			case BIT:
			case SIGN:
				unsigned char* vectorData = (unsigned char*)(data);
				bitCopyTo(vectorData, interface);
				break;
			}
		}

	public:
		virtual ImplementationType getImplementationType()
		{
			return SSE2;
		};

		virtual VectorType getVectorType()
		{
			return vectorTypeTempl;
		};

		XmmVector()
		{
		}
		;

		XmmVector(unsigned size)
		{
			this->tSize = size;

			size_t byteSize = getByteSize(size, vectorTypeTempl);
			data = mi_malloc(byteSize);

			switch (vectorTypeTempl)
			{

			case BYTE:
				SetValueToAnArray<unsigned char> (data, byteSize, 128);
				break;
			case FLOAT:
				SetValueToAnArray<float> (data, byteSize / sizeof(float), 0);
				break;
			case BIT:
			case SIGN:
				SetValueToAnArray<unsigned char> (data, byteSize, 0);
				break;
			}
		}

		~XmmVector()
		{
			if (data) {
				mi_free(data);
				data = NULL;
			}
		}

		virtual Vector* clone()
		{
			Vector* clone = new XmmVector<vectorTypeTempl, c_typeTempl> (tSize);
			copyTo(clone);
			return clone;
		}

		virtual void activation(Vector* resultsVect, FunctionType functionType)
		{
			float* results = (float*)resultsVect->getDataPointer();

			switch (vectorTypeTempl)
			{
			case BYTE:
			{
				std::string error =
						"XmmVector::activation is not implemented for VectorType BYTE.";
				throw error;
			}
				break;
			case FLOAT:
			{
				for (unsigned i = 0; i < tSize; i++) {
					((c_typeTempl*)data)[i] = Function<c_typeTempl> (
							results[i], functionType);
				}
			}
				break;
			case BIT:
			case SIGN:
			{
				unsigned char* vectorData = (unsigned char*)data;

				unsigned blockOffset = 0;
				unsigned bytePos = 0;
				unsigned char vectorMask = 128;

				for (unsigned i = 0; i < tSize; i++) {

					if (results[i] > 0) {
						vectorData[blockOffset + bytePos] |= vectorMask;
					}
					else {
						vectorData[blockOffset + bytePos] &= ~vectorMask;
					}

					if (i % BYTES_PER_BLOCK == (BYTES_PER_BLOCK - 1)) {
						bytePos = 0;
						if (i % BITS_PER_BLOCK == (BITS_PER_BLOCK - 1)) {
							blockOffset += BYTES_PER_BLOCK;
							vectorMask = 128;
						}
						else {
							vectorMask >>= 1;
						}
					}
					else {
						++bytePos;
					}
				}
			}
			}
		}
	};

#endif /* XMMVECTOR_H_ */
