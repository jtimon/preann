
#include "chronometer.h"

Chronometer::Chronometer()
{
	start_time = -1;
	end_time = -1;
	timeInSeconds = 0;
}

Chronometer::~Chronometer()
{
}

void Chronometer::start()
{
	if (end_time != -1){
		cout<<"Warning: the chronometer was already started."<<endl;
	} else {
		timeInSeconds = 0;
		start_time = clock();
	}
}

void Chronometer::stop()
{
	if (start_time == -1) {
		std::string error = "The chronometer must be started before stop it.";
		throw error;
	} else {
		end_time = clock();
		timeInSeconds = (end_time-start_time)/(double)CLOCKS_PER_SEC;
		start_time = -1;
		end_time = -1;
	}
}

float Chronometer::getSeconds()
{
	return timeInSeconds;
}
