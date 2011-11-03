//////////////////////////////////////////////////////////////////////////////*

     Archivo    :    tablero.h
     Descripci¢n:    Implementacion del TAD Tablero de la practica REVERSI
     Otros datos:
		     Elementos de Programacion. Curso 03/04

     Fecha de creacion :    7-01-2004
     Ultima modificacion:   13-01-2004

*/////////////////////////////////////////////////////////////////////////////

#include "casilla.h"

//////////////////////////Definicion del Tipo///////////////////////////////

	       
 typedef struct Tablero {
      casilla m[Max][Max];
      int tam;
    };

 
//////////////Declaracion de Funciones y Variables/////////////////////////

int fichas1, fichas2, x, y;
register int i, j;
boolean enc;


tablero crearT (int tam);
void plantarT (Tablero &tab);
int cuantasFichasT (Tablero tab, casilla jugador);
boolean encerrarT (Tablero tab, casilla jugador, int x, int y);
void actualizarT (Tablero &tab, casilla jugador, int x, int y);
boolean hayMovimientoPosibleT (Tablero tab, casilla jugador);
boolean finPartidaT (Tablero tab);

///*nuevos modulos*///

void mostrarPuntuacion (tablero tab);
void rellenarRejilla (tablero tab);
void turno (tablero &tab, casilla jugador); 
int tamaniotablero (tablero tab);

//////////////////////////Definicion de Modulos////////////////////////////


    tablero crearT (int tam) {
        
        tablero tab;
        for (i=0; i<tam; i++) {
           for (j=0; j<tam; j++) {
              tab.m[i][j]=crearC();
           }
        } 
        tab.tam=tam;
    }


    void plantarT (Tablero &tab) {

          ocuparC (tab.m[tab.tam/2][tab.tam/2], jugador1);
          ocuparC (tab.m[tab.tam/2-1][tab.tam/2-1], jugador1);
          ocuparC (tab.m[tab.tam/2-1][tab.tam/2], jugador2);
          ocuparC (tab.m[tab.tam/2][tab.tam/2-1], jugador2);
    }


    int cuantasFichasT (Tablero tab, casilla jugador) {
          
          int fichas=0;
          for (i=0; i<tab.tam; i++) {
             for (j=0; j<tab.tam; j++) {
                if (tab.m[i][j]==jugador) {
                   fichas++;
             }
          } 
    }


    boolean encerrarT (Tablero tab, casilla jugador, int x, int y) {

               enc=falso;
              if(tab[x][y]==vacia) {
		 for (a=-1; a<2; a++){    //para cada una de las 8 posibilidades(arriba,abajo
	            for(b=-1; b<2; b++){  /*izquierda, arribaizquierda...) mas una novena*/
                                         /*que descarta   3*3=9  */
	               i=x+a; j=y+b;     
	               while (i>=0 && j>=0 && i<tab.tam && j<tab.tam) {

                          if (enc==cierto) {break;}  /*no es preciso que siga buscando*/

                          if (a==0 && b==0) {break;} /*aqui descarta la 9ªposibilidad*/

		          if (tab[i][j]==contrario(jugador)) {
			      i+=a; j+=b;   /*como es del adversario, sigue avanzando*/
                          }                  /*en la misma direccion    */
                                               
		          else {
			      if (tablero[i][j]==jugador){ 
                                 for(i-=a; i<x; i-=a) {enc=cierto;}
                              }
                             i=-1; //sea del mismo jugador o este vacia, ha de salir del bucle
		          }/*else*/
			 
		       }/*while*/
		    }/*for de b*/
		  }  /*for de a*/
                } /*primer if*/
		 return enc;
     }


    void actualizarT (Tablero &tab, casilla jugador, int x, int y) {

           if(tablero[x][y]==vacia) { /*funciona como encerrar*/

		 for (a=-1; a<2; a++){    
	            for(b=-1; b<2; b++){  
                                         
	               i=x+a; j=y+b;     
	               while (i>=0 && j>=0 && i<tab.tam && j<tab.tam) {
                         
                          if (a==0 && b==0) {break;} 
		          if (tab[i][j]==contrario(jugador)) {
			      i+=a; j+=b;   
                          }                  
                                               
		          else {
			      if (tab[i][j]==jugador){ 
                                 for(i-=a; i<x; i-=a) {
                                     girarC (tab[i][j]);
                                 }
                              }
                              i=-1; 
		          }/*else*/
			 
		       }/*while*/
		    }/*for de b*/
		 }   /*for a*/
                 girarc (tab[x][y]);
           } /*if*/
    }


    boolean hayMovimientoPosibleT (Tablero tab, casilla jugador) {

          enc=falso;
          for (i=0; i<tab.tam; i++) {
             for (j=0; j<tab.tam; j++) {
                if (encerrarT (tab[i][j], jugador)==cierto) {
                    enc==cierto;
                }
                if (enc==cierto) {break;} /*es suficiente con una jugada posible*/
             }
             if (enc==cierto) {break;}
          } 
          return enc;
    }


    boolean finPartidaT (Tablero tab) {

             enc=falso;
       
             if (hayMovimientoPosibleT (tab, jugador1)==falso &&
                 hayMovimientoPosibleT (tab, jugador2)==falso) {
                  enc=cierto;
             }
    } 

    
    void mostrarPuntuacion (tablero tab) {

           fichas1=cuantasFichasT (tab, jugador1);
           fichas2=cuantasFichasT (tab, jugador2);
           mensajePuntos (fichas1, fichas2);
    }


    void rellenarRejilla (tablero tab) {

           for (i=0; i<tab.tam; i++) {
              for (j=0; j<tab.tam; j++) {
                 pintarFicha (i, j, tab.m[i][j]);
              }
           }           
    }


    void turno (tablero tab, casilla jugador) {

          if (hayMovimientoPosibleT (tab, jugador)==cierto) {
               mensajeTurno ("posesión del turno:", estadoC (jugador));
               movercursor (tab.tam, x, y);
               if (encerrarT (tab, jugador, x, y)) {
                    actualizarT (tab, jugador, x, y);
               }
               rellenarRejilla (tab);
               mostrarPuntuacion (tab);
          }
          else {
               mensaje ("no puedes mover: se pasa el turno");               
    }


    int tamaniotablero (tablero tab) {

         return tab.tam;
    }
  
