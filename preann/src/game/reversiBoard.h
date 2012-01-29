/*
 * reversiBoard.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef REVERSIBOARD_H_
#define REVERSIBOARD_H_

#include "board.h"

class ReversiBoard : public Board
{
public:
    ReversiBoard(ReversiBoard* other);
    ReversiBoard(unsigned size);
    virtual ~ReversiBoard();

    virtual void initBoard();

    virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual bool canMove(SquareState player);
    virtual bool endGame();
    virtual void turn(SquareState player, Individual* individual = NULL);
    virtual float computerEstimation(unsigned xPos, unsigned yPos,
            SquareState player);
    virtual float individualEstimation(unsigned xPos, unsigned yPos,
            SquareState player, Individual* individual);

};

#endif /* REVERSIBOARD_H_ */
