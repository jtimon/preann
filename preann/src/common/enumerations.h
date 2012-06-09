/*
 * enumerations.h
 *
 *  Created on: Sep 11, 2011
 *      Author: timon
 */

#include "util.h"

#ifndef ENUMERATIONS_H_
#define ENUMERATIONS_H_

typedef enum
{
    BT_FLOAT, BT_BIT, BT_SIGN, BT_BYTE
} BufferType;
#define BUFFER_TYPE_DIM 4
typedef enum
{
    FT_IDENTITY, FT_BINARY_STEP, FT_BIPOLAR_STEP, SIGMOID, FT_BIPOLAR_SIGMOID, FT_HYPERBOLIC_TANGENT
} FunctionType;
#define FUNCTION_TYPE_DIM 6
typedef enum
{
    IT_C, IT_SSE2, IT_CUDA, IT_CUDA_REDUC, IT_CUDA_INV
} ImplementationType;
#define IMPLEMENTATION_TYPE_DIM 5
typedef enum
{
    SA_ROULETTE_WHEEL, SA_RANKING, SA_TOURNAMENT, SA_TRUNCATION
} SelectionAlgorithm;
#define SELECTION_ALGORITHM_DIM 4
typedef enum
{
    CL_WEIGH, CL_NEURON, CL_NEURON_INVERTED, CL_LAYER
} CrossoverLevel;
#define CROSSOVER_LEVEL_DIM 4
typedef enum
{
    CA_UNIFORM, CA_PROPORTIONAL, CA_MULTIPOINT
} CrossoverAlgorithm;
#define CROSSOVER_ALGORITHM_DIM 3
typedef enum
{
    MA_DISABLED, MA_PER_INDIVIDUAL, MA_PROBABILISTIC
} MutationAlgorithm;
#define MUTATION_ALGORITHM_DIM 3
typedef enum
{
    RA_DISABLED, RA_PER_INDIVIDUAL, RA_PROBABILISTIC
} ResetAlgorithm;
#define RESET_ALGORITHM_DIM 3
typedef enum
{
    BO_OR, BO_AND, BO_XOR
} BinaryOperation;
#define BINARY_OPERATION_DIM 3
typedef enum
{
    TT_BIN_OR, TT_BIN_AND, TT_BIN_XOR, TT_REVERSI
} TestTask;
#define TEST_TASKS_DIM 4

typedef enum
{
    ET_BUFFER, ET_IMPLEMENTATION, ET_FUNCTION, ET_SELECTION_ALGORITHM, ET_CROSS_LEVEL, ET_CROSS_ALG,
    ET_MUTATION_ALG, ET_RESET_ALG, ET_BINARY_OPERATION, ET_TEST_TASKS
} EnumType;
#define ENUM_TYPE_DIM 10

class Enumerations
{
private:
    Enumerations()
    {
    }
    ;
    virtual ~Enumerations()
    {
    }
    ;
public:
    static unsigned enumTypeDim(EnumType enumType);
    static std::string toString(EnumType enumType, unsigned enumValue);
    static std::string crossoverLevelToString(unsigned crossoverLevel);
    static std::string selectionAlgorithmToString(unsigned selectionAlgorithm);
    static std::string crossoverAlgorithmToString(unsigned crossoverAlgorithm);
    static std::string mutationAlgorithmToString(unsigned mutationAlgorithm);
    static std::string resetAlgorithmToString(unsigned resetAlgorithm);
    static std::string bufferTypeToString(unsigned bufferType);
    static std::string implementationToString(unsigned implementationType);
    static std::string enumTypeToString(EnumType enumType);
    static std::string functionTypeToString(unsigned functionType);
    static std::string binaryOperationToString(unsigned binaryOperation);
    static std::string testTaskToString(unsigned testTask);
};

#endif /* ENUMERATIONS_H_ */
