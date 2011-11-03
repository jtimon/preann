////////////////////////////////////////////////////////////////////////////*

     Archivo    :    casilla.h
     Descripcion:    Implementacion del TAD Casilla de la practica REVERSI
     Otros datos:
		     Elementos de Programacion. Curso 03/04

     Fecha de creacion :    6-01-2004
     Ultima modificacion:   13-01-2004

*///////////////////////////////////////////////////////////////////////////

#include "entorno.h"

////////////////definición del tipo de dato/////////////////////////////////


typedef enum casilla {vacia, jugador1, jugador2};


//////////////////Declaracion de Funciones/////////////////////////////////

  	
 Casilla crearC ();
 void ocuparC (Casilla c, casilla jugador);
 void girarC (Casilla c);
 int estadoC (Casilla c);
 void pintarC (Casilla c, int x, int y);

 casilla contraria (casilla c);

/////////////////Definicion de los modulos/////////////////////////////////


  Casilla crearC (){

	Casilla c;
  	c = vacia;
  	return c;
  }


  void ocuparC (Casilla &c, casilla jugador){

  	c = jugador;
  }

 
  void girarC (Casilla &c){

  	if (c == jugador1){	
  	   c=jugador2;
	}
  	else {
  	   if (c == jugador2){		
  	      c=jugador1;
	   }	
  }


  int estadoC (Casilla c){

        int x;
        if (c==jugador1) { x=1; }
        if (c==jugador2) { x=2; }
        else             { x=0; }
	return x;
  }


  void pintarC (Casilla c, int x, int y){ 
 	
 	pintarFicha(x, y, estadoC (c)); 			
  }

  
  casilla contraria (casilla c) {

          if (c == jugador1){	
  	      c=jugador2;
	  }
  	  if (c == jugador2){		
  	       c=jugador1;
	  }
          return c;
   }
	

