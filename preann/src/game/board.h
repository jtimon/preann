/*
 * board.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef BOARD_H_
#define BOARD_H_

#include "genetic/individual.h"

typedef enum
{
    EMPTY, PLAYER_1, PLAYER_2
} SquareState;

struct Move
{
    unsigned xPos;
    unsigned yPos;
    float quality;
};

class Board
{
protected:
    unsigned tSize;
    SquareState** tBoard;
    Interface* tInterface;
public:
    Board(unsigned size);
    virtual ~Board();

    virtual void initBoard();
    virtual bool endGame() = 0;
    virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
    virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
    virtual bool canMove(SquareState player) = 0;
    virtual void turn(SquareState player, Individual* individual = NULL) = 0;
    virtual float computerEstimation(unsigned xPos, unsigned yPos,
            SquareState player) = 0;
    virtual float individualEstimation(unsigned xPos, unsigned yPos,
            SquareState player, Individual* individual) = 0;
    virtual unsigned countPoints(SquareState player);

    unsigned getSize();
    void setSquare(unsigned xPos, unsigned yPos, SquareState squareState);
    SquareState getSquare(unsigned xPos, unsigned yPos);
    Interface* getInterface();
    Interface* updateInterface();
    static SquareState opponent(SquareState player);
};

#endif /* BOARD_H_ */
