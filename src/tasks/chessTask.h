/*
 * chessTask.h
 *
 * Chess task for training neural networks through gameplay
 */

#ifndef CHESS_TASK_H_
#define CHESS_TASK_H_

#include "genetic/task.h"
#include "game/chessBoard.h"

class ChessTask : public Task
{
private:
    ChessBoard* tBoard;
    unsigned tNumTests;
    Individual* tBestOpponent;  // Best individual to use as opponent (competitive co-evolution)

public:
    ChessTask(BufferType bufferType, unsigned numTests = 1);
    virtual ~ChessTask();

    virtual void test(Individual* individual);
    virtual void setInputs(Individual* individual);
    virtual std::string toString();
    virtual Individual* getExample(ParametersMap* parameters);
    virtual float getGoal();

    // Competitive co-evolution: manage best opponent
    void setBestOpponent(Individual* opponent);
    Individual* getBestOpponent();
    bool hasStoredOpponent();
};

#endif /* CHESS_TASK_H_ */
