#ifndef GOTASK_H_
#define GOTASK_H_

#include "genetic/task.h"
#include "game/goBoard.h"

/**
 * GoTask - Task for training neural networks to play Go
 *
 * This task evaluates individuals by having them play Go against
 * a computer opponent. Follows the same pattern as ReversiTask.
 *
 * Note: Currently not functional - requires implementation of
 * Go game rules (via Fuego library or custom implementation).
 */
class GoTask : public Task
{
    GoBoard* tBoard;
    unsigned tNumTests;

public:
    GoTask(unsigned size, BufferType bufferType, unsigned numTests = 1);
    virtual ~GoTask();

    virtual void test(Individual* individual);
    virtual void setInputs(Individual* individual);
    virtual std::string toString();
    virtual Individual* getExample(ParametersMap* parameters);
    virtual float getGoal();
};

#endif /* GOTASK_H_ */
