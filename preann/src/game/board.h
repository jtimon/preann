/*
 * board.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef BOARD_H_
#define BOARD_H_

#include "interface.h"

typedef enum {EMPTY, PLAYER_1, PLAYER_2} SquareState;

struct Move {
	unsigned xPos;
	unsigned yPos;
	float quality;
};

class Board {
	unsigned tSize;
	SquareState** tBoard;
	Interface* tInterface;
public:
	Board(unsigned size);
	virtual ~Board();
	
	virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
	virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player) = 0;
	virtual bool canMove(SquareState player) = 0;
	virtual bool endGame() = 0;
	virtual void computerTurn(SquareState turn) = 0;
	
	unsigned size();
	void setSquare(unsigned xPos, unsigned yPos, SquareState squareState);
	SquareState getSquare(unsigned xPos, unsigned yPos);
	Interface* getInterface();
	void updateInterface();
	unsigned countSquares(SquareState squareState);
	static SquareState opponent(SquareState player);
};

#endif /* BOARD_H_ */
