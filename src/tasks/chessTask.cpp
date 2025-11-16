/*
 * chessTask.cpp
 *
 * Chess task implementation
 */

#include "chessTask.h"
#include "game/board.h"
#include "common/util.h"

using namespace std;

ChessTask::ChessTask(BufferType bufferType, unsigned numTests)
{
    tBoard = new ChessBoard(8, bufferType);
    tNumTests = numTests;
}

ChessTask::~ChessTask()
{
    delete tBoard;
}

void ChessTask::test(Individual* individual)
{
    float fitness = 0.0;

    for (unsigned i = 0; i < tNumTests; ++i) {
        // Alternate which player the individual plays (fairness)
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;

        tBoard->initBoard();
        SquareState turn = PLAYER_1;  // White always starts
        unsigned moveCount = 0;
        unsigned maxMoves = 200;  // Prevent infinite games

        while (!tBoard->endGame() && moveCount < maxMoves) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    // Neural network plays
                    tBoard->turn(turn, individual);
                } else {
                    // Computer heuristic plays
                    tBoard->turn(turn, NULL);
                }
                moveCount++;
            }
            turn = Board::opponent(turn);
        }

        // Calculate fitness based on win/loss/draw
        // Chess has no "points", just outcomes
        if (tBoard->isCheckmate(Board::opponent(individualPlayer))) {
            fitness += 100.0;  // Win
        } else if (tBoard->isCheckmate(individualPlayer)) {
            fitness += 0.0;    // Loss
        } else {
            fitness += 50.0;   // Draw/stalemate/timeout
        }
    }

    // Average fitness across all games
    individual->setFitness(fitness / tNumTests);
}

void ChessTask::setInputs(Individual* individual)
{
    // Configure neural network input layer to match chess board
    // 768 inputs (8x8x12 for piece-aware encoding)
    individual->addInputLayer(tBoard->getInterface());
}

Individual* ChessTask::getExample(ParametersMap* parameters)
{
    // Create example neural network architecture for chess
    // Input: 768 neurons (8x8 squares × 12 piece types)
    // Hidden: 16-16 (small/fast architecture as requested)
    // Output: 1 (position evaluation)

    ImplementationType implementationType;
    BufferType bufferType;
    FunctionType functionType;

    implementationType = (ImplementationType)parameters->getNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    bufferType = (BufferType)parameters->getNumber(Enumerations::enumTypeToString(ET_BUFFER));
    functionType = (FunctionType)parameters->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));

    Individual* example = new Individual(implementationType);
    this->setInputs(example);  // 768 input neurons

    // Hidden layers
    example->addLayer(16, bufferType, functionType);
    example->addLayer(16, bufferType, functionType);

    // Output layer
    example->addLayer(1, BT_FLOAT, functionType);

    // Connections
    example->addInputConnection(0, 0);   // Input to first hidden layer
    example->addLayersConnection(0, 1);  // First to second hidden layer
    example->addLayersConnection(1, 2);  // Second hidden to output

    return example;
}

float ChessTask::getGoal()
{
    // Goal fitness = consistently winning
    // Win = 100, Draw = 50, Loss = 0
    // Goal: average of 90+ (winning most games)
    return 90.0;
}

string ChessTask::toString()
{
    return "CHESS";
}
