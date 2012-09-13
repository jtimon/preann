/**
* \file chronometer.h
*   \brief Cabecera de la clase Chronometer.
*   \author
*        \b Jorge Timón Morillo-Velarde\n
*/

#ifndef CHRONOMETER_H_
#define CHRONOMETER_H_

using namespace std;

#include <iostream>

#ifdef CUDA_IMPL
#include "cuda/cuda.h"
#endif

/** \class Chronometer
*   \brief clase para cronometrar el rendimiento de los métodos a optimizar
*
*   Produce resultados poco fiables cuando cronometra tiempos muy largos o cuando hay entrada/salida involucrada.
*/
class Chronometer
{
protected:
    clock_t start_time;
    //!<almacena el valor del reloj del sistema cuando se inicia la cuenta con start.
    clock_t end_time;
    //!<almacena el valor del reloj del sistema cuando se detiene la cuenta con stop.
    float timeInSeconds;
    //!<almacena el tiempo que se ha tardado (en segundos) cuando se detiene la cuenta con stop.

public:
    Chronometer();
    virtual ~Chronometer();

    void start();
    void stop();
    float getSeconds();
};

#endif /* CHRONOMETER_H_ */
