/*
 * chessTask.cpp
 *
 * Chess task implementation
 */

#include "chessTask.h"
#include "game/board.h"
#include "common/util.h"
#include <fstream>
#include <sstream>

using namespace std;

ChessTask::ChessTask(BufferType bufferType, unsigned numTests, bool enableGameLogging)
{
    tBoard = new ChessBoard(8, bufferType);
    tNumTests = numTests;
    tBestOpponent = NULL;  // Start with random opponent (bootstrap)
    tEnableGameLogging = enableGameLogging;
    tCounterFile = "data/chess_game_counter.txt";
    tGameCounter = 0;

    if (tEnableGameLogging) {
        loadGameCounter();
    }
}

ChessTask::~ChessTask()
{
    delete tBoard;
    if (tBestOpponent != NULL) {
        delete tBestOpponent;
    }
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
        unsigned maxMoves = 400;  // Prevent infinite games

        while (!tBoard->endGame() && moveCount < maxMoves) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    // Neural network being tested plays
                    tBoard->turn(turn, individual);
                } else {
                    // Opponent plays: use stored best opponent if available, else random
                    // Avoid self-play: if testing the best opponent itself, use random
                    Individual* opponent = (tBestOpponent != NULL && tBestOpponent != individual)
                                          ? tBestOpponent : NULL;
                    tBoard->turn(turn, opponent);
                }
                moveCount++;
            }
            turn = Board::opponent(turn);
        }

        // Calculate fitness based on win/loss/draw
        // Chess has no "points", just outcomes
        bool isCheckmate = false;
        bool individualWon = false;

        if (tBoard->isCheckmate(Board::opponent(individualPlayer))) {
            fitness += 1.0;  // Win
            isCheckmate = true;
            individualWon = true;
        } else if (tBoard->isCheckmate(individualPlayer)) {
            fitness += -1.0;    // Loss
            isCheckmate = true;
            individualWon = false;
        } else {
            fitness += 0.0;   // Draw/stalemate/timeout
        }

        // Log interesting games (checkmate wins/losses only, not draws)
        if (tEnableGameLogging && isCheckmate) {
            Individual* opponent = (tBestOpponent != NULL && tBestOpponent != individual)
                                  ? tBestOpponent : NULL;
            logGame(individual, individualPlayer, opponent, moveCount, individualWon);
        }
    }

    // Total fitness across all games (not averaged)
    // Max fitness = tNumTests (winning all games)
    // Min fitness = -tNumTests (losing all games)
    individual->setFitness(fitness);
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
    // Goal fitness = winning all games
    // Win = +1, Draw = 0, Loss = -1
    // Goal: tNumTests (winning all tNumTests games)
    return (float)tNumTests;
}

string ChessTask::toString()
{
    return "CHESS";
}

void ChessTask::setBestOpponent(Individual* opponent)
{
    // Delete old copy if exists
    if (tBestOpponent != NULL) {
        delete tBestOpponent;
    }

    // Make a copy of the opponent so we own it (population may delete original)
    tBestOpponent = (opponent != NULL) ? opponent->newCopy(true) : NULL;
}

Individual* ChessTask::getBestOpponent()
{
    return tBestOpponent;
}

bool ChessTask::hasStoredOpponent()
{
    return tBestOpponent != NULL;
}

void ChessTask::loadGameCounter()
{
    ifstream counterFile(tCounterFile);
    if (counterFile.is_open()) {
        counterFile >> tGameCounter;
        counterFile.close();
    } else {
        // File doesn't exist, start from 0
        tGameCounter = 0;
    }
}

void ChessTask::saveGameCounter()
{
    ofstream counterFile(tCounterFile);
    if (counterFile.is_open()) {
        counterFile << tGameCounter << endl;
        counterFile.close();
    }
}

void ChessTask::logGame(Individual* ind, SquareState indPlayer, Individual* opp, unsigned moves, bool indWon)
{
    // Increment and save counter
    tGameCounter++;
    saveGameCounter();

    // Create filename
    ostringstream filename;
    filename << "output/games/chess/game" << tGameCounter << ".txt";

    // Write game file
    ofstream gameFile(filename.str().c_str());
    if (!gameFile.is_open()) {
        return;  // Silently fail if can't create file
    }

    // Header
    gameFile << "=== Chess Game #" << tGameCounter << " ===" << endl;
    gameFile << "Individual: " << (indPlayer == PLAYER_1 ? "PLAYER_1 (White)" : "PLAYER_2 (Black)") << endl;

    // Opponent type
    if (opp != NULL) {
        gameFile << "Opponent: Best Individual";
        if (opp == ind) {
            gameFile << " (self-play fallback to random)";
        }
    } else {
        gameFile << "Opponent: Random";
    }
    gameFile << endl;

    // Result
    if (indWon) {
        gameFile << "Result: " << (indPlayer == PLAYER_1 ? "PLAYER_1" : "PLAYER_2") << " wins by checkmate" << endl;
    } else {
        SquareState winner = Board::opponent(indPlayer);
        gameFile << "Result: " << (winner == PLAYER_1 ? "PLAYER_1" : "PLAYER_2") << " wins by checkmate" << endl;
    }

    gameFile << "Moves: " << moves << endl;
    gameFile << endl;

    // Piece legend for humans
    gameFile << "White pieces: P R N B Q K (uppercase)" << endl;
    gameFile << "Black pieces: p r n b q k (lowercase)" << endl;
    gameFile << endl;

    // Final board position
    gameFile << "Final board position:" << endl;
    tBoard->printBoard(gameFile);

    gameFile.close();
}
