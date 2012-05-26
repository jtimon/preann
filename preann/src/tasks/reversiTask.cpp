/*
 * reversiTask.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiTask.h"

ReversiTask::ReversiTask(unsigned size, unsigned numTests)
{
    tBoard = new ReversiBoard(size);
    tNumTests = numTests;
}

ReversiTask::~ReversiTask()
{
    delete (tBoard);
}

float ReversiTask::getGoal()
{
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
                    tBoard->turn(turn, NULL);
                }
            }
//            tBoard->print();
            turn = Board::opponent(turn);
        }
        //TODO controlar negativos ??
        fitness += tBoard->countPoints(individualPlayer);
//        tBoard->print();
//        cout << " points " << tBoard->countPoints(individualPlayer) << endl;
    }
    individual->setFitness(fitness);
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

