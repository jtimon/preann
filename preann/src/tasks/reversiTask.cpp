/*
 * reversiTask.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiTask.h"

ReversiTask::ReversiTask(unsigned size, unsigned numTests)
{
	board = new ReversiBoard(size);
	tNumTests = numTests;
}

ReversiTask::~ReversiTask()
{
	// TODO Auto-generated destructor stub
}

void ReversiTask::test(Individual* individual)
{
	float fitness = 0;
	for (unsigned i = 0; i < tNumTests; ++i) {

		SquareState turn = PLAYER_1;
		board->initBoard();

		while(!board->endGame()){
			if(board->canMove(turn)){
				// this way the individual moves first half of the games
				if(i % 2 == 0){
					individualTurn(turn, individual);
					board->computerTurn(turn);
				} else {
					board->computerTurn(turn);
					individualTurn(turn, individual);
				}
			}
			turn = Board::opponent(turn);
		}
	}
	individual->setFitness(fitness);
}

void ReversiTask::setInputs(Individual* individual)
{
	individual->addInputLayer(board->getInterface());
}

std::string ReversiTask::toString()
{
	return "REVERSI_" + to_string(board->size());
}

void ReversiTask::individualTurn(SquareState turn, Individual* individual)
{
	//TODO ReversiTask::individualTurn modificar con lo que corresponda
	float maxQuality = 0;
	vector<Move> moves;
	for (int x = 0; x < board->size(); ++x) {
		for (int y = 0; y < board->size(); ++y) {
			Move move;
			move.xPos = x;
			move.yPos = y;
			move.quality = (float)board->getQuality(x, y, turn);
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
		board->makeMove(chosenMove.xPos, chosenMove.yPos, turn);
	}
}

