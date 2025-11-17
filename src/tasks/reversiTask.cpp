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
    unsigned totalGames = 0;

    // When adversary exists: play 2 games vs neural adversary + 2 games vs greedy
    // When no adversary: play 2 games vs greedy only
    bool hasNeuralAdversary = (tAdversary != NULL);

    // Play 2 games vs greedy computer (bootstrap baseline)
    for (unsigned i = 0; i < 2; ++i) {
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;
        SquareState turn = PLAYER_1;
        tBoard->initBoard();

        while (!tBoard->endGame()) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    tBoard->turn(turn, individual);
                } else {
                    // Play against greedy computer (NULL = greedy strategy)
                    tBoard->turn(turn, NULL);
                }
            }
            turn = Board::opponent(turn);
        }

        float individualPoints = tBoard->countPoints(individualPlayer);
        float opponentPoints = tBoard->countPoints(Board::opponent(individualPlayer));
        fitness += individualPoints - opponentPoints;
        totalGames++;
    }

    // If neural adversary exists, play 2 additional games vs adversary
    if (hasNeuralAdversary) {
        for (unsigned i = 0; i < 2; ++i) {
            SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;
            SquareState turn = PLAYER_1;
            tBoard->initBoard();

            while (!tBoard->endGame()) {
                if (tBoard->canMove(turn)) {
                    if (turn == individualPlayer) {
                        tBoard->turn(turn, individual);
                    } else {
                        // Play against neural adversary
                        tBoard->turn(turn, tAdversary);
                    }
                }
                turn = Board::opponent(turn);
            }

            float individualPoints = tBoard->countPoints(individualPlayer);
            float opponentPoints = tBoard->countPoints(Board::opponent(individualPlayer));
            fitness += individualPoints - opponentPoints;
            totalGames++;
        }
    }

    individual->setFitness(fitness / totalGames);
}

void ReversiTask::testBootstrap(Individual* individual)
{
    // Test only against greedy computer (8 games)
    float fitness = 0;

    for (unsigned i = 0; i < 8; ++i) {
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;
        SquareState turn = PLAYER_1;
        tBoard->initBoard();

        while (!tBoard->endGame()) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    tBoard->turn(turn, individual);
                } else {
                    // Play against greedy computer (NULL = greedy strategy)
                    tBoard->turn(turn, NULL);
                }
            }
            turn = Board::opponent(turn);
        }

        float individualPoints = tBoard->countPoints(individualPlayer);
        float opponentPoints = tBoard->countPoints(Board::opponent(individualPlayer));
        fitness += individualPoints - opponentPoints;
    }

    individual->setFitness(fitness / 8.0);
}

void ReversiTask::testAdversary(Individual* individual)
{
    // Test only against neural adversary (2 games)
    // Requires tAdversary to be set
    if (tAdversary == NULL) {
        individual->setFitness(0);
        return;
    }

    float fitness = 0;

    for (unsigned i = 0; i < 2; ++i) {
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;
        SquareState turn = PLAYER_1;
        tBoard->initBoard();

        while (!tBoard->endGame()) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    tBoard->turn(turn, individual);
                } else {
                    // Play against neural adversary
                    tBoard->turn(turn, tAdversary);
                }
            }
            turn = Board::opponent(turn);
        }

        float individualPoints = tBoard->countPoints(individualPlayer);
        float opponentPoints = tBoard->countPoints(Board::opponent(individualPlayer));
        fitness += individualPoints - opponentPoints;
    }

    individual->setFitness(fitness / 2.0);
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

