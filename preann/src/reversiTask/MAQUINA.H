//////////////////////////////////////////////////////////////////////////////*

     Archivo    :    maquina.h
     Descripci¢n:    Implementacion del TAD Tablero de la practica REVERSI
     Otros datos:
		     Elementos de Programacion. Curso 03/04

     Fecha de creacion :    12-01-2004
     Ultima modificacion:   13-01-2004

*/////////////////////////////////////////////////////////////////////////////

#include "tablero.h"

//////////////////////////Definicion del Tipo y constantes///////////////////


 const int dificultadMax = 25;
 const int dificultadMin = 1;
	       
 typedef struct jugada {
      
      int x; int y;
      int calidad;
    };
 typedef jugada vectorjugada[dificultadMax];

 
//////////////Declaracion de Funciones y Variables/////////////////////////

 register int k;
 jugada jug;

 jugada crearJugada (tablero tab, int x, int y);
 void llenarVector (tablero tab, vectorjugadas &v);
 jugada elegirJugada (vectorjugada v, int dificultad); 
 void turnoMaquina (tablero &tab, int dificultad);

//////////////////////////Definicion de Modulos////////////////////////////


   jugada crear (tablero tab, int x, int y) {
       
         if(tablero[x][y]==0) { /*funciona como encerrarT*/

                 jug.x = x;
                 jug.y = y;
                 jug.calidad = 1;

		 for (a=-1; a<2; a++){    
	            for(b=-1; b<2; b++){  
                                         
	               i=x+a; j=y+b;     
	               while (i>=0 && j>=0 && i<tab.tam && j<tab.tam) {
                         
                          if (a==0 && b==0) {break;} 
		          if (tab.m[i][j]==contrario(jugador)) {
			      i+=a; j+=b;   
                          }                  
                                               
		          else {
			      if (tab.m[i][j]==jugador){ 
                                 for(i-=a; i<x; i-=a) {
                                     jug.calidad++;
                                 }
                              }
                              i=-1; 
		          }/*else*/
			 
		       }/*while*/
		    }/*for de b*/
		 }   /*for a*/                
           } /*if*/
           return jug;   
   }


   void llenarVector (tablero tab, vectorjugada &v) {
        
        k=0;
        for (i=0; i<tab.tam; i++) {
             for (j=0; j<tab.tam; j++) {
                if (encerrarT (tab, jugador2, i, j)==cierto) {
                    v[k]=crearJugada (tab, i, j);
                    k++;
                }  
             }
        } 
   }

   
   jugada elegirJugada (vectorjugada v, int dificultad) {

        k=0;
        for (i=1; i<dif; i++) {
            if ( v[k].calidad < v[i].calidad ) {
               k=i;
            }
        }   
   }


   void turnoMaquina (tablero &tab, int dificultad) {

		vectorJugadas v;
		
        if (hayMovimientoPosibleT (tab, jugador2)==cierto) {

               mensajeTurno ("posesión del turno:", 2);
		getch();
   
	       crearVector (tab, v);
               jug = elegirJugada (v , dificultad);               
               x = sacarCoordenadax (jug);
               y = sacarCoordenaday (jug);

               actualizarT (tab, jugador, x, y);
               
               rellenarRejilla (tab);
               mostrarPuntuacion (tab);
          }
          else {
               mensaje ("no puede mover: se pasa el turno");
		getch();
          }
    }   