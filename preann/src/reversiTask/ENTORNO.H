/*********************************************************************
     Archivo    :    entorno.h
     Descripci¢n:    Implementacion del entorno de la practica REVERSI
     Otros datos:
		     Elementos de Programacion. Curso 03/04

     Fecha de creacion :    5-12-2003
     Ultima modificacion:

 *********************************************************************/
#include <conio.h>

// Declaracion de constantes. Limites del tamaño del tablero
const int Min   = 4;
const int Max   = 10;

//Colores de las fichas de los jugadores
const int Ficha_JUGADOR1 = LIGHTBLUE;
const int Ficha_JUGADOR2 = LIGHTGREEN;
const int Ficha_VACIA    = BLACK;



// Tipos usuales
const int MAXCARACTERES = 80;
typedef enum boolean   {falso,cierto};
typedef char tipoCadena [MAXCARACTERES];




/*-----------------------------------------------------------------
   Descripcion:
	Dibuja la rejilla del tablero en la pantalla.
	Necesita las dimensiones del tablero que es cuadrado y
	y viene indicado por tamanio.

   Precondicion:
	Tamanio debe estar entre las constantes Min y Max

 -----------------------------------------------------------------*/
void pintarRejilla (int tamanio);




/*--------------------------------------------------------------------
   Descripcion:
	   Dibuja una ficha en la casilla indicada por x e y  del color
	   indicado

   Precondicion:
	   x (columna) toma valores entre 0 y (tamanio - 1);
		       de izquierda a derecha
	   y (fila) toma valores entre 0 y (tamanio - 1);
		       de arriba a abajo
	   color debe tomar el valor de alguna de las dos constantes
	      Ficha_JUGADOR1 o Ficha_JUGADOR2
 ---------------------------------------------------------------------*/
void pintarFicha (int x, int y, int color);




/*---------------------------------------------------------------------
   Descripcion:
	   Resalta la casilla en la que se encuentra el cursor

   Precondicion:
	   x (columna) toma valores entre 0 y (tamanio - 1);
		       de izquierda a derecha
	   y (fila) toma valores entre 0 y (tamanio - 1);
		       de arriba a abajo
 ----------------------------------------------------------------------*/
void resaltarCasilla (int x, int y);




/*---------------------------------------------------------------------*
   Descripcion:
	   Borra el resaltado de la casilla que ha abandonado el cursor

   Precondicion:
	   x (columna) toma valores entre 0 y (tamanio - 1);
		       de izquierda a derecha
	   y (fila) toma valores entre 0 y (tamanio - 1);
		       de arriba a abajo
 ----------------------------------------------------------------------*/
void borrarCasilla (int tamanio, int x, int y);




/*---------------------------------------------------------------------
   Descripcion:
	   Muestra un mensaje para leer un valor entero comprendido
	   entre los valores (min..max)

   Precondicion:
	   min < max
	   c es el mensaje que se que se muestra
   Postcondicion:
	   entero esta en el rango [min, max]


 ---------------------------------------------------------------------*/
int leerEntero (tipoCadena c, int min, int max);




/*---------------------------------------------------------------------
   Descripcion:
	   Muestra un mensaje en la pantalla
	   c es la cadena de caracteres que se muestra

   Precondicion:
	   C no es la cadena vacia

 ---------------------------------------------------------------------*/
void mensaje (tipoCadena c);


/*---------------------------------------------------------------------
   Descripcion:
	   Muestra un mensaje en la pantalla que contiene un dato numerico
	   c es la cadena de caracteres que se muestra
	   numero es el dato numerico

   Precondicion:
	   C no es la cadena vacia
 ---------------------------------------------------------------------*/

void mensajeConNumero (tipoCadena c, int numero);


/*---------------------------------------------------------------------
   Descripcion:
	   Muestra un mensaje en la pantalla para mostrar las fichas 
           ocupadas por los dos jugadores
	   FichasJugador1: es el número de fichas del jugador 1
	   FichasJugador2: es el número de fichas del jugador 2
   
 ---------------------------------------------------------------------*/

void mensajePuntos (int FichasJugador1, int FichasJugador2);



/*---------------------------------------------------------------------
   Descripcion:
	   Muestra un mensaje en la pantalla para indicar el turno del
	   jugador
	   c: es el mensaje que se muestra
	   jugador: es el jugador al que le toca

   Precondicion:
	   Jugador es 1 o 2

 ---------------------------------------------------------------------*/


void mensajeTurno (tipoCadena c, int jugador);



/*---------------------------------------------------------------------
   Descripcion:
	Coloca el cursor en el centro del tablero, y permite que
	rste se mueva libremente mediante los cursores, hasta que
	se pulse la tecla ENTER, momento en el que devuelve las
	coordenadas de la posicion del cursor.

   Precondici¢n:
	tamanio: Min..Max

   Postcondicion:
	x: 1..Max
	y: 1..Max
 ---------------------------------------------------------------------*/
void moverCursor (int tamanio, int &x, int&y);

