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
};

#endif /* GOBOARD_H_ */
