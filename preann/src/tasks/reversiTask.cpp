/*
 * reversiTask.cpp
 *
 *  Created on: 04/11/2011
 *      Author: jtimon
 */

#include "reversiTask.h"

ReversiTask::ReversiTask(unsigned size, unsigned numTests) 
{
	board = new Board(size);
	tNumTests = numTests;
}

ReversiTask::~ReversiTask() 
{
	// TODO Auto-generated destructor stub
}

virtual void ReversiTask::test(Individual* individual)
{
	
}

void ReversiTask::setInputs(Individual* individual)
{
	
}

std::string ReversiTask::toString()
{
	return "REVERSI_" + to_string(board.size());
}
