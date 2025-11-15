#include "goTask.h"

using namespace std;

GoTask::GoTask(unsigned size, BufferType bufferType, unsigned numTests)
{
    tBoard = new GoBoard(size, bufferType);
    tNumTests = numTests;
}

GoTask::~GoTask()
{
    delete tBoard;
}

void GoTask::test(Individual* individual)
{
    // Calculate fitness by playing multiple games
    // This follows the same pattern as ReversiTask
    //
    // NOTE: This will fail at runtime because GoBoard methods are not implemented
    // Once Go rules are implemented (via Fuego or custom), this will work

    unsigned maxPoints = tBoard->getSize() * tBoard->getSize();
    float fitness = 0;

    for (unsigned i = 0; i < tNumTests; ++i) {
        // Alternate who plays first (black vs white)
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;
        SquareState turn = PLAYER_1;
        tBoard->initBoard();

        // Play the game
        while (!tBoard->endGame()) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    // Neural network player
                    tBoard->turn(turn, individual);
                } else {
                    // Computer opponent (simple heuristic)
                    tBoard->turn(turn, NULL);
                }
            }
            turn = Board::opponent(turn);
        }

        // Add points to fitness (maxPoints offset to keep fitness positive)
        fitness += tBoard->countPoints(individualPlayer) + maxPoints;
    }

    // Set average fitness across all test games
    individual->setFitness(fitness / tNumTests);
}

void GoTask::setInputs(Individual* individual)
{
    // Set up the neural network input layer
    // Input is the board state: size * size positions
    individual->addInputLayer(tBoard->getInterface());
}

string GoTask::toString()
{
    return "GoTask";
}

Individual* GoTask::getExample(ParametersMap* parameters)
{
    // Create an example neural network for Go
    // Following the same pattern as ReversiTask

    BufferType bufferType;
    try {
        bufferType = (BufferType) parameters->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    } catch (string& e) {
        bufferType = BT_BIT;
    }
    ImplementationType implementationType;
    try {
        implementationType = (ImplementationType) parameters->getNumber(
                Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    } catch (string& e) {
        implementationType = IT_C;
    }
    FunctionType functionType;
    try {
        functionType = (FunctionType) parameters->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));
    } catch (string& e) {
        functionType = FT_IDENTITY;
    }

    unsigned boardSize = tBoard->getSize();
    Individual* example = new Individual(implementationType);
    this->setInputs(example);
    example->addLayer(boardSize, bufferType, functionType);
    example->addLayer(boardSize, bufferType, functionType);
    example->addLayer(1, BT_FLOAT, functionType);
    example->addInputConnection(0, 0);
    example->addLayersConnection(0, 1);
    example->addLayersConnection(0, 2);

    return example;
}

float GoTask::getGoal()
{
    // Goal fitness for Go
    // In Go, scoring is more complex than Reversi due to territory counting
    // For now, use a similar goal to Reversi (most of the board)
    //
    // This will need adjustment once actual Go scoring is implemented

    unsigned maxPoints = tBoard->getSize() * tBoard->getSize();
    return maxPoints * 1.8;  // Goal: win by controlling ~80% of board
}
