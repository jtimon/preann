/*
 * reversiBoard.h
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#ifndef REVERSIBOARD_H_
#define REVERSIBOARD_H_

#include "board.h"

class ReversiBoard: public Board {
public:
	ReversiBoard(unsigned size);
	virtual ~ReversiBoard();
	
	void initBoard();
	
	virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player);
	virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player);
	virtual bool canMove(SquareState player);
	virtual bool endGame();
};

#endif /* REVERSIBOARD_H_ */
