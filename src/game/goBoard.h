#ifndef GOBOARD_H_
#define GOBOARD_H_

#include "board.h"

/**
 * GoBoard - Implementation of the ancient game of Go
 *
 * Following the same pattern as ReversiBoard, this class provides
 * a Go game implementation for use with PREANN's genetic algorithm
 * neural network training.
 *
 * Standard Go board sizes: 9x9, 13x13, 19x19
 */
class GoBoard : public Board
{
private:
    // Ko rule: store previous board state hash to detect immediate recapture
    unsigned long long tPreviousBoardHash;

    // Pass tracking: game ends when both players pass consecutively
    unsigned consecutivePasses;

    // Helper functions for Go rules implementation
    void findGroup(unsigned xPos, unsigned yPos, SquareState player,
                   bool visited[][19], std::vector<std::pair<unsigned, unsigned>>& group);
    int countLiberties(unsigned xPos, unsigned yPos, SquareState player);
    void removeGroup(const std::vector<std::pair<unsigned, unsigned>>& group);
    bool wouldCapture(unsigned xPos, unsigned yPos, SquareState player);
    unsigned long long calculateBoardHash();

public:
    GoBoard(GoBoard* other);
    GoBoard(unsigned size, BufferType bufferType);
    virtual ~GoBoard();

    virtual void initBoard();

    // Pure virtual methods from Board base class
    virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual float computerEstimation(unsigned xPos, unsigned yPos, SquareState player);
    virtual float individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual);

    // Override turn() to add passing as a move option
    virtual void turn(SquareState player, Individual* individual = NULL);

    // Override endGame() to check for two consecutive passes
    virtual bool endGame();

    // Pass move - increments consecutive pass counter
    void pass();
};

#endif /* GOBOARD_H_ */
