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

    bool insideBoard(int xPos, int yPos);
private:
    Board(){};
    void baseConstructor(unsigned size, BufferType bufferType);
public:
    Board(unsigned size, BufferType bufferType);
    Board(Board* other);
    virtual ~Board();

    virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
    virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
    virtual float computerEstimation(unsigned xPos, unsigned yPos, SquareState player) = 0;
    virtual float individualEstimation(unsigned xPos, unsigned yPos, SquareState player,
                                       Individual* individual) = 0;

    virtual void initBoard();
    virtual bool canMove(SquareState player);
    virtual void turn(SquareState player, Individual* individual = NULL);
    virtual bool endGame();
    virtual int countPoints(SquareState player);

    unsigned getSize();
    void setSquare(unsigned xPos, unsigned yPos, SquareState squareState);
    SquareState getSquare(unsigned xPos, unsigned yPos);
    bool squareIs(int xPos, int yPos, SquareState squareState);
    Interface* getInterface();
    Interface* updateInterface();
    static SquareState opponent(SquareState player);
    void print();
    BufferType getBufferType();
};

#endif /* BOARD_H_ */
