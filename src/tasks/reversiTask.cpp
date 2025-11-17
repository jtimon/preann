/*
 * reversiTask.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiTask.h"

ReversiTask::ReversiTask(unsigned size, BufferType bufferType, unsigned numTests)
{
    tBoard = new ReversiBoard(size, bufferType);
    tNumTests = numTests;
    tAdversary = NULL;  // Start with random opponent (bootstrap)
}

ReversiTask::~ReversiTask()
{
    delete (tBoard);
    if (tAdversary != NULL) {
        delete tAdversary;
    }
}

float ReversiTask::getGoal()
{
    // Goal: win all games with perfect score (64-0 = +64 per game)
    // With tNumTests games, maximum fitness = tNumTests * 64
    return tNumTests * tBoard->getSize() * tBoard->getSize();
}

void ReversiTask::test(Individual* individual)
{
    float fitness = 0;
    for (unsigned i = 0; i < tNumTests; ++i) {

        SquareState individualPlayer;
        // this way the individual moves first only half of the games
        if (i % 2 == 0) {
            individualPlayer = PLAYER_1;
        } else {
            individualPlayer = PLAYER_2;
        }

        SquareState turn = PLAYER_1;
        tBoard->initBoard();

        while (!tBoard->endGame()) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    tBoard->turn(turn, individual);
                } else {
                    // Opponent plays: use fixed adversary if available, else random
                    tBoard->turn(turn, tAdversary);
                }
            }
            turn = Board::opponent(turn);
//            tBoard->print();
        }
        // Fitness based on score difference: your points - opponent's points
        // Positive = win, negative = loss, zero = tie
        float individualPoints = tBoard->countPoints(individualPlayer);
        float opponentPoints = tBoard->countPoints(Board::opponent(individualPlayer));
        fitness += individualPoints - opponentPoints;
//        tBoard->print();
//        cout << " points " << tBoard->countPoints(individualPlayer) << endl;
    }
    individual->setFitness(fitness/tNumTests);
//    cout << "fitness " << fitness << endl;
}

void ReversiTask::setInputs(Individual* individual)
{
    individual->addInputLayer(tBoard->getInterface());
}

Individual* ReversiTask::getExample(ParametersMap* parameters)
{
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

std::string ReversiTask::toString()
{
    return "REVERSI_" + to_string(tBoard->getSize());
}

void ReversiTask::setAdversary(Individual* adversary)
{
    // Delete old copy if exists
    if (tAdversary != NULL) {
        delete tAdversary;
    }

    // Make a copy of the adversary so we own it (population may delete original)
    tAdversary = (adversary != NULL) ? adversary->newCopy(true) : NULL;
}

Individual* ReversiTask::getAdversary()
{
    return tAdversary;
}

bool ReversiTask::hasAdversary()
{
    return tAdversary != NULL;
}

