/*
 * reversiBoard.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiBoard.h"

ReversiBoard::ReversiBoard(unsigned size) : Board(size) 
{
	std::string error;
	if (size % 2 != 0){
		error = "The size of a Reversi board must be even.";
		throw error;
	}
	if (size < 4){
		error = "The minimum size of a Reversi board is 4.";
		throw error;
	}
}

ReversiBoard::~ReversiBoard() 
{
}

void ReversiBoard::initBoard()
{
    int halfSize = tSize / 2;
    setSquare(halfSize, halfSize, PLAYER_1);
	setSquare(halfSize-1, halfSize-1, PLAYER_1);
	setSquare(halfSize-1, halfSize, PLAYER_2);
	setSquare(halfSize, halfSize-1, PLAYER_2);
}

bool ReversiBoard::legalMove(unsigned xPos, unsigned yPos, SquareState player)
{
	if (xPos >= tSize || yPos >= tSize){
		std::string error = "ReversiBoard::legalMove : The position (" + to_string(xpos) + ", " +
				to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
		throw error;
	}
	if (player == EMPTY){
		std::string error = "ReversiBoard::legalMove : Empty square is not a player.";
		throw error;
	}
	if (tBoard[xPos][yPos] != EMPTY){
		std::string error = "ReversiBoard::legalMove : the square at position (" + to_string(xpos) +
				", " + to_string(yPos) + ") is already accupied";
		throw error;
	}
	unsigned x=0, y=0;
	
	for (a=-1; a<2; a++){    //for each direction (left, right)
		for(b=-1; b<2; b++){  // for each direction (up down)
			if (a==0 && b==0) {
				break;
			} else {
				x = xPos + a;
				y = yPos + b;     
			}
			//at least one of the squares has to belong to the opponent
			if (tBoard[x][y] == opponent(player)) {
				//up to the borders
				while (x >= 0 && y >= 0 && x < tSize && y < tSize) {
				
					if (tBoard[x][y] == opponent(player)) {
						x+=a; y+=b;
					} else if (tBoard[x][y] == player){
						return true;
					}
				}
			}
		}
	}
	return false;
}

void ReversiBoard::makeMove(unsigned xPos, unsigned yPos, SquareState player)
{
	if (xPos >= tSize || yPos >= tSize){
		std::string error = "ReversiBoard::makeMove : The position (" + to_string(xpos) + ", " +
				to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
		throw error;
	}
	if (player == EMPTY){
		std::string error = "ReversiBoard::makeMove : Empty square is not a player.";
		throw error;
	}
	if (tBoard[xPos][yPos] != EMPTY){
		std::string error = "ReversiBoard::makeMove : the square at position (" + to_string(xpos) +
				", " + to_string(yPos) + ") is already accupied";
		throw error;
	}
	unsigned x=0, y=0;
	
	for (a=-1; a<2; a++){    //for each direction (left, right)
		for(b=-1; b<2; b++){  // for each direction (up down)
			if (a==0 && b==0) {
				break;
			} else {
				x = xPos + a;
				y = yPos + b;     
			}
			//at least one of the squares has to belong to the opponent
			if (tBoard[x][y] == opponent(player)) {
				//up to the borders
				while (x >= 0 && y >= 0 && x < tSize && y < tSize) {
				
					if (tBoard[x][y] == opponent(player)) {
						x += a; 
						y += b;
					} else if (tBoard[x][y] == player){
						
						x -= a; 
						y -= b;
						// set the squares inbetween to the player
						while (x != xPos && y != yPos){
							tBoard[x][y] = player;
							x -= a; 
							y -= b;
						}
						break;
					}
				}
			}
		}
	}
}

bool ReversiBoard::canMove(SquareState player)
{
	if (player == EMPTY){
		std::string error = "ReversiBoard::canMove : Empty square is not a player.";
		throw error;
	}
	for (int x = 0; x < tSize; ++x) {
		for (int y = 0; y < tSize; ++y) {
			
			if(legalMove(x, y, player)){
				return true;
			}
		}
	}
	return false;
}

bool ReversiBoard::endGame()
{
	if (canMove(PLAYER_1) || canMove(PLAYER_2)){
		return false;
	}
	return true;
}

void ReversiBoard::computerTurn(SquareState turn)
{
	float maxQuality = 0;
	vector<Move> moves;
	for (int x = 0; x < tSize; ++x) {
		for (int y = 0; y < tSize; ++y) {
			Move move;
			move.xPos = x;
			move.yPos = y;
			move.quality = (float)getQuality(x, y, turn);
			if (move.quality >= maxQuality){
				maxQuality = move.quality; 
				moves.push_back(move);
			}
		}
	}
	vector<Move> bestMoves;
	for (int i = 0; i < moves.size(); ++i) {
		if (moves[i].quality == maxQuality){
			bestMoves.push_back(moves[i]);
		}
	}
	if (bestMoves.size() > 0){
		Move chosenMove = bestMoves[ Random::positiveInteger(bestMoves.size()) ];
		makeMove(chosenMove.xPos, chosenMove.yPos, turn);
	}
}

unsigned ReversiBoard::getQuality(unsigned xPos, unsigned yPos, SquareState player)
{
	if (xPos >= tSize || yPos >= tSize){
		std::string error = "ReversiBoard::legalMove : The position (" + to_string(xpos) + ", " +
				to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
		throw error;
	}
	if (player == EMPTY){
		std::string error = "ReversiBoard::legalMove : Empty square is not a player.";
		throw error;
	}
	if (tBoard[xPos][yPos] != EMPTY){
		std::string error = "ReversiBoard::legalMove : the square at position (" + to_string(xpos) +
				", " + to_string(yPos) + ") is already accupied";
		throw error;
	}
	unsigned x=0, y=0;
	unsigned quality = 0;
	
	for (a=-1; a<2; a++){    //for each direction (left, right)
		for(b=-1; b<2; b++){  // for each direction (up down)
			if (a==0 && b==0) {
				break;
			} else {
				x = xPos + a;
				y = yPos + b;     
			}
			//at least one of the squares has to belong to the opponent
			if (tBoard[x][y] == opponent(player)) {
				//up to the borders
				while (x >= 0 && y >= 0 && x < tSize && y < tSize) {
				
					if (tBoard[x][y] == opponent(player)) {
						x+=a; y+=b;
					} else if (tBoard[x][y] == player){
						x -= a; 
						y -= b;
						// count quality
						while (x != xPos && y != yPos){
							++quality;
							x -= a; 
							y -= b;
						}
						break;
					}
				}
			}
		}
	}
	return quality;
}
