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
    tAdversary = NULL;  // Start with random opponent (bootstrap)
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
    if (tAdversary != NULL) {
        delete tAdversary;
    }
}

void ChessTask::test(Individual* individual)
{
    float fitness = 0.0;

    for (unsigned i = 0; i < tNumTests; ++i) {
        // Alternate which player the individual plays (fairness)
        SquareState individualPlayer = (i % 2 == 0) ? PLAYER_1 : PLAYER_2;

        tBoard->initBoard();

        // Reset recurrent layer state at the start of each game
        // This gives the network a clean slate to accumulate state across moves
        // Layer 2 (SIGN/bipolar) will start at all zeros (interpreted as -1)
        individual->getOutput(2)->reset();

        SquareState turn = PLAYER_1;  // White always starts
        unsigned moveCount = 0;
        unsigned maxMoves = 400;  // Prevent infinite games

        while (!tBoard->endGame() && moveCount < maxMoves) {
            if (tBoard->canMove(turn)) {
                if (turn == individualPlayer) {
                    // Neural network being tested plays
                    tBoard->turn(turn, individual);
                } else {
                    // Opponent plays: use fixed adversary if available, else random
                    tBoard->turn(turn, tAdversary);
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
            Individual* opponent = (tAdversary != NULL && tAdversary != individual)
                                  ? tAdversary : NULL;
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
    // Architecture: 768→128(BIT)→128(BIT)→32(SIGN)→1(FLOAT) with recurrent connection
    // Strategy: Make binary layers bigger (memory efficient with byte weights),
    //           keep bipolar layer smaller for efficient output layer reading
    // Recurrent: Layer 2 (SIGN) feeds back to layer 0, providing memory across moves
    //           in a single game to potentially learn opponent modeling

    ImplementationType implementationType;
    FunctionType functionType;

    implementationType = (ImplementationType)parameters->getNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    functionType = (FunctionType)parameters->getNumber(Enumerations::enumTypeToString(ET_FUNCTION));

    Individual* example = new Individual(implementationType);
    this->setInputs(example);  // 768 input neurons

    // ===== SIMPLE DEBUG NETWORK (for fast testing) =====
    // Uncomment for quick benchmarks and debugging
    // Architecture: 768→1→1→1→1 (minimal single-neuron layers)
    // Has layer 2 so reset() works without commenting
    example->addLayer(1, BT_BIT, functionType);      // Layer 0
    example->addLayer(1, BT_BIT, functionType);      // Layer 1
    example->addLayer(1, BT_SIGN, functionType);     // Layer 2 (for reset compatibility)
    example->addLayer(1, BT_FLOAT, FT_IDENTITY);     // Layer 3: output

    example->addInputConnection(0, 0);               // Input (768) → Layer 0 (1)
    example->addLayersConnection(0, 1);              // Layer 0 (1) → Layer 1 (1)
    example->addLayersConnection(1, 2);              // Layer 1 (1) → Layer 2 (1)
    example->addLayersConnection(2, 3);              // Layer 2 (1) → output (1)
    // No recurrent connection in debug network (not needed for benchmarking)

    // ===== PRODUCTION NETWORK (comment out for debugging) =====
    // Uncomment for actual training and evolution
    // Architecture: 768→128→128→32→1 with recurrent connection
    /*
    // Hidden layers - using BIT buffer for first two layers (byte weights)
    example->addLayer(128, BT_BIT, functionType);    // Layer 0
    example->addLayer(128, BT_BIT, functionType);    // Layer 1

    // Third hidden layer - SIGN buffer for bipolar values (also byte weights, smaller for efficiency)
    // This layer will feed back to layer 0, creating recurrent memory
    example->addLayer(32, BT_SIGN, functionType);    // Layer 2 (recurrent)

    // Output layer - FLOAT with IDENTITY function for position evaluation
    example->addLayer(1, BT_FLOAT, FT_IDENTITY);     // Layer 3

    // Feedforward connections
    example->addInputConnection(0, 0);   // Input (768) to first hidden (128)
    example->addLayersConnection(0, 1);  // First hidden (128) to second hidden (128)
    example->addLayersConnection(1, 2);  // Second hidden (128) to third hidden (32)
    example->addLayersConnection(2, 3);  // Third hidden (32) to output (1)

    // Recurrent connection - layer 2 feeds back to layer 0
    // This provides memory across moves within a single game
    example->addLayersConnection(2, 0);  // Recurrent: bipolar (32) → first hidden (128)
    */

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

void ChessTask::setAdversary(Individual* adversary)
{
    // Delete old copy if exists
    if (tAdversary != NULL) {
        delete tAdversary;
    }

    // Make a copy of the adversary so we own it (population may delete original)
    tAdversary = (adversary != NULL) ? adversary->newCopy(true) : NULL;
}

Individual* ChessTask::getAdversary()
{
    return tAdversary;
}

bool ChessTask::hasAdversary()
{
    return tAdversary != NULL;
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
