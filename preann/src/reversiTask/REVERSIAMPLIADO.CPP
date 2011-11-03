//////////////////////////////////////////////////////////////////////*
     Archivo    :    reversiampliado.cpp
     Descripcion:    Implementacion del programa principal
			 de la practica REVERSI
     Otros datos:
		     Elementos de Programacion. Curso 03/04

     Fecha de creacion :    7-01-2004
     Ultima modificacion:   13-01-2004

*/////////////////////////////////////////////////////////////////////

#include "maquina.h"

////////////////////Programa Principal////////////////////////

    void main() {


      casilla turnoActivo; int tamanio, maquina; tablero tab;

      tamanio = leerEntero ("Tamaño del tablero", Min, Max);
      maquina = leerEntero ("¿jugar contra la maquina?(1 para aceptar, 0 para cancelar)", 0, 1);
      dificultad = leerEntero ("Seleccione la dificultad (de 1 a 25)", dificultadMin, dificultadMax);

	X = tamanio / 2;
	Y = tamanio / 2;
	pintarRejilla (tamanio);
	tab = crearT (tamanio);
	plantarT (tab);
       	turnoActivo = jugador1;
        
	      while (finPartidaT (tab)==falso) {
		
                   turno (tab, turnoActivo);
                   turnomaquina (tab, dificultad);
		   if (maquina != 1) {
                       girarC (turnoActivo);
                   }
	}


        if (fichas1>fichas2) { 
               mensajeturno ("ha ganado el ", 1); }
        else { mensajeturno ("ha ganado el ", 2); }
   
    }
