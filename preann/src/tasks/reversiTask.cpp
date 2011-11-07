/*
 * reversiTask.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiTask.h"

ReversiTask::ReversiTask(unsigned size, unsigned numTests)
{
	tBoard = new ReversiBoard(size);
	tNumTests = numTests;
}

ReversiTask::~ReversiTask()
{
	delete(tBoard);
}

void ReversiTask::test(Individual* individual)
{
	float fitness = 0;
	for (unsigned i = 0; i < tNumTests; ++i) {

		SquareState individualPlayer;
		// this way the individual moves first only half of the games
		if(i % 2 == 0){
			individualPlayer = PLAYER_1;
		} else {
			individualPlayer = PLAYER_2;
		}
		
		SquareState turn = PLAYER_1;
		tBoard->initBoard();
			
		while(!tBoard->endGame()){
			if(tBoard->canMove(turn)){
				if (turn == individualPlayer){
					tBoard->turn(turn, individual);
				} else {
					tBoard->turn(turn, NULL);
				}
			}
			turn = Board::opponent(turn);
		}
		//TODO controlar negativos ??
		fitness += tBoard->countPoints(individualPlayer);
	}
	individual->setFitness(fitness);
}

void ReversiTask::setInputs(Individual* individual)
{
	individual->addInputLayer(tBoard->getInterface());
}

std::string ReversiTask::toString()
{
	return "REVERSI_" + to_string(tBoard->size());
}
