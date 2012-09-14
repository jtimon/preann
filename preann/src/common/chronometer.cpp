/** \file chronometer.cpp
*   \brief Implemetación de los métodos de la clase Chronometer.
*   \author
*        \b Jorge Timón Morillo-Velarde\n
*/

#include "chronometer.h"

/** Constructor por defecto de la clase Chronometer.
*   \param " " este método no recibe parámetros
*   \return No devuelve nada
*/
Chronometer::Chronometer()
{
    start_time = -1;
    end_time = -1;
    timeInSeconds = 0;
}

/** Destructor de la clase Chronometer.
*   \param " " este método no recibe parámetros
*   \return No devuelve nada
*/
Chronometer::~Chronometer()
{
}

/** Inicia la cuenta del cronómetro.
*   \param " " este método no recibe parámetros
*   \return No devuelve nada
*/
void Chronometer::start()
{
#ifdef CUDA_IMPL
        cuda_synchronize();
#endif
    if (end_time != -1) {
        cout << "Warning: the chronometer was already started." << endl;
    } else {
        timeInSeconds = 0;
        start_time = clock();
    }
}

/** Detiene la cuenta del cronómetro.
*   \param " " este método no recibe parámetros
*   \return No devuelve nada
*/
void Chronometer::stop()
{
#ifdef CUDA_IMPL
        cuda_synchronize();
#endif

    if (start_time == -1) {
        std::string error = "The chronometer must be started before stop it.";
        throw error;
    } else {
        end_time = clock();
        timeInSeconds = (end_time - start_time) / (double)CLOCKS_PER_SEC;
        start_time = -1;
        end_time = -1;
    }
}

/** Devuelve la cuenta almacenada en el cronómetro. Se debe haber llamado antes a Chronometer::stop().
*   \return Devuelve el número de segundos (con decimales) en la cuenta.
*/
float Chronometer::getSeconds()
{
    return timeInSeconds;
}
