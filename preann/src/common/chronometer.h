#ifndef CHRONOMETER_H_
#define CHRONOMETER_H_

using namespace std;

#include <iostream>

class Chronometer
{
protected:
    clock_t start_time;
    clock_t end_time;
    float timeInSeconds;

public:
    Chronometer();
    virtual ~Chronometer();

    void start();
    void stop();
    float getSeconds();
};

#endif /* CHRONOMETER_H_ */
