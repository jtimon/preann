/*
 * board.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "board.h"

Board::Board(unsigned size)
{
	tSize = size;
	tBoard = (SquareState**)MemoryManagement::malloc(sizeof(SquareState) * tSize * tSize);
	tInterface = new Interface(tSize * tSize * 2, BT_BIT);
}

Board::~Board()
{
	MemoryManagement::free(tBoard);
	delete(tInterface);
}

unsigned Board::size()
{
	return tSize;
}

void Board::setSquare(unsigned xPos, unsigned yPos, SquareState squareState)
{
	if (xPos >= tSize || yPos >= tSize){
		std::string error = "Board::setSquare : The position (" + to_string(xPos) + ", " +
				to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
		throw error;
	}
	tBoard[xPos][yPos] = squareState;
}

SquareState Board::getSquare(unsigned xPos, unsigned yPos)
{
	if (xPos >= tSize || yPos >= tSize){
		std::string error = "Board::getSquare : The position (" + to_string(xPos) + ", " +
				to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
		throw error;
	}
	return tBoard[xPos][yPos];
}

Interface* Board::getInterface()
{
	return tInterface;
}

void Board::updateInterface()
{
	unsigned index = 0;
	for (int x = 0; x < tSize; ++x) {
		for (int y = 0; y < tSize; ++y) {

			float play1=0, play2=0;
			if(tBoard[x][y] == PLAYER_1){
				play1 = 1;
			} else if(tBoard[x][y] == PLAYER_2){
				play2 = 1;
			}
			tInterface->setElement(index++, play1);
			tInterface->setElement(index++, play2);
		}
	}
}

unsigned Board::countPoints(SquareState player)
{
	unsigned points = 0;
	for (int x = 0; x < tSize; ++x) {
		for (int y = 0; y < tSize; ++y) {
			if (tBoard[x][y] == player){
				++points;
			} else if (tBoard[x][y] == opponent(player)){
				--points;
			}
		}
	}
	return points;
}

SquareState Board::opponent(SquareState player)
{
	if (player == EMPTY){
		std::string error = "Board::adversary : Empty square is not a player.";
		throw error;
	}
	if(player == PLAYER_1){
		return PLAYER_2;
	}
	if(player == PLAYER_2){
		return PLAYER_1;
	}
}

