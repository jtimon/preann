section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

section .text use32

global XMMbinario2
XMMbinario2: 

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH EDX
	PUSH ESI
	PUSH EDI

	MOV EDX, [ESP + 28]  ;numIndicesEntradas
	MOV EDI, [ESP + 32]  ;pesos
	MOV EAX, 0

	PXOR XMM3, XMM3      ;anulamos el registro donde iremos acumulando las sumas
	PXOR XMM0, XMM0      ;anulamos otro registro que usaremos cuando necesitemos
	MOVDQU XMM6, [mascara];iniciamos la mascara para cuando la tengamos que usar

bucleVector:

	MOV ESI, [ESP+24]    ;vectorIndicesEntradas
	ADD ESI, EAX

	MOV ECX, [ESI+4]     
	SHR ECX, 4           ;dividimos entre 16 (el número de entradas que se procesan por vuelta del bucle)

	MOV ESI, [ESI]

	MOV BL, 1	      ;para que la primera vez se inicie la mascara 
bucle:

	PSRLW XMM4, 1        ;desplazamos las mascaras de entradas
	DEC BL               ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara ;si BL==0, se terminó de procesar el bloque de entrada		

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV BL, 8             ;ciclos por bloque de entrada	
	MOVDQU XMM1, [ESI]    ;leemos el bloque de estados de entrada que toca
	ADD ESI, 16           ;actualizamos el puntero

noIniciarMascara:

	MOVDQA XMM7, XMM4     ;copiamos la mascacara en una mascara auxiliar

	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM5, XMM5    ;ponemos todo XMM5 a 1
	PXOR XMM7, XMM5	      ;invertimos XMM7 (ahora está todo el byte a 1 en los bytes que tenian el bit que tocaba activo)

	MOVDQU XMM5, XMM6     ;128 en todos los bytes de XMM5  
	PAND XMM5, XMM7       ;128 en todos los bytes que estaban activos

	MOVDQU XMM2, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos
	PAND XMM7, XMM2       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit activo

	PSADBW XMM7, XMM0     ;sumamos todos los bytes (los que estaban activos)
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba activo

	PADDD XMM3, XMM7      ;sumamos estos pesos a los ya sumados previamente
	PSUBD XMM3, XMM5      ;sustraemos 128 por cada bit que estaba activo

	DEC ECX
	JNZ bucle

	ADD EAX, 8            ;le sumamos a EAX el tamaño de una estructura de las que hay en el vector
	DEC EDX
	JNZ bucleVector

	MOVD EAX, XMM3        ;copiamos la mitad baja del resultado
	PSRLDQ XMM3, 8        ;desplazamos el registro 8 bytes a la derecha
	MOVD ECX, XMM3        ;copiamos la mitad alta del resultado
	ADD EAX, ECX          ;sumamos ambas mitades

	;restauramos los registros
	POP EDI
	POP ESI
	POP EDX
	POP ECX
	POP EBX

	RET


global XMMbipolar2
XMMbipolar2: 

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH EDX
	PUSH ESI
	PUSH EDI

	MOV EDX, [ESP + 28]  ;numIndicesEntradas
	MOV EDI, [ESP + 32]  ;pesos
	MOV EAX, 0

	MOVDQU XMM6, [mascara];iniciamos la mascara para cuando la tengamos que usar
	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él
	PXOR XMM0, XMM0       ;anulamos XMM0 para cuando lo tengamos que usar

bucleVector2:

	MOV ESI, [ESP+24]    ;vectorIndicesEntradas
	ADD ESI, EAX

	MOV ECX, [ESI+4]     
	SHR ECX, 4           ;dividimos entre 16 (el número de entradas que se procesan por vuelta del bucle)

	MOV ESI, [ESI]

	MOV BL, 1	      ;para que la primera vez se inicie la mascara 
bucle2:

	PSRLW XMM4, 1         ;desplazamos las mascaras de entradas
	DEC BL                ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara2 ;si BL==0, se terminó de procesar el bloque de entrada		

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV BL, 8             ;ciclos por bloque de entrada	
	MOVDQU XMM1, [ESI]    ;leemos el bloque de estados de entrada que toca
	ADD ESI, 16           ;actualizamos el puntero

noIniciarMascara2:


	MOVDQA XMM7, XMM4     ;copiamos la mascacara en una mascara auxiliar
	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM2, XMM2    ;ponemos a 1 todo el XMM2
	PXOR XMM2, XMM7	      ;en XMM2 esta el inverso de XMM7 


	MOVDQU XMM5, XMM6     ;128 en todo XMM5
	PAND XMM5, XMM2       ;128 en todos los bytes que estaban activos
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba activo
	PSUBD XMM3, XMM5      ;sustraemos 128 por cada bit que estaba activo
	
	MOVDQU XMM5, XMM6     ;128 en todo XMM5
	PAND XMM5, XMM7       ;128 en todos los bytes que estaban inactivos
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba inactivo
	PADDD XMM3, XMM5      ;sumamos 128 por cada bit que estaba inactivo


	MOVDQU XMM5, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos
	PAND XMM2, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit activo
	PAND XMM7, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit inactivo


	PSADBW XMM2, XMM0     ;sumamos todos los bytes (los que estaban activos)
	PSADBW XMM7, XMM0     ;sumamos todos los bytes (los que estaban inactivos)

	PADDD XMM3, XMM2      ;sumamos los pesos "positivos" al acumulador
	PSUBD XMM3, XMM7      ;sustraemos los pesos "negativos" al acumulador

	DEC ECX
	JNZ bucle2

	ADD EAX, 8            ;le sumamos a EAX el tamaño de una estructura de las que hay en el vector
	DEC EDX
	JNZ bucleVector2

	MOVD EAX, XMM3        ;copiamos la mitad baja del resultado
	PSRLDQ XMM3, 8        ;desplazamos el registro 8 bytes a la derecha
	MOVD EDX, XMM3        ;copiamos la mitad alta del resultado
	ADD EAX, EDX          ;sumamos ambas mitades

	;restauramos los registros
	POP EDI
	POP ESI
	POP EDX
	POP ECX
	POP EBX

	RET

;extern "C" int XMMbinario (void* vectorEntrada, unsigned numeroBloques, unsigned char* pesos);
global XMMbinario
XMMbinario: 

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 20]  ;vector de entradas
	MOV ECX, [ESP + 24]  ;numero de entradas
	;SHR ECX, 4           ;dividimos entre 16 (el número de entradas que se procesan por vuelta del bucle)
	MOV EDI, [ESP + 28]  ;pesos

	PXOR XMM3, XMM3      ;anulamos el registro donde iremos acumulando las sumas
	PXOR XMM0, XMM0      ;anulamos otro registro que usaremos cuando necesitemos
	MOVDQU XMM6, [mascara];iniciamos la mascara para cuando la tengamos que usar
	MOV BL, 1	      ;para que la primera vez se inicie la mascara 
bucle3:

	PSRLW XMM4, 1        ;desplazamos las mascaras de entradas
	DEC BL               ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara3;si BL==0, se terminó de procesar el bloque de entrada		

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV BL, 8             ;ciclos por bloque de entrada	
	MOVDQU XMM1, [ESI]    ;leemos el bloque de estados de entrada que toca
	ADD ESI, 16           ;actualizamos el puntero

noIniciarMascara3:

	MOVDQA XMM7, XMM4     ;copiamos la mascacara en una mascara auxiliar

	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM5, XMM5    ;ponemos todo XMM5 a 1
	PXOR XMM7, XMM5	      ;invertimos XMM7 (ahora está todo el byte a 1 en los bytes que tenian el bit que tocaba activo)

	MOVDQU XMM5, XMM6     ;128 en todos los bytes de XMM5  
	PAND XMM5, XMM7       ;128 en todos los bytes que estaban activos

	MOVDQU XMM2, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos
	PAND XMM7, XMM2       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit activo

	PSADBW XMM7, XMM0     ;sumamos todos los bytes (los que estaban activos)
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba activo

	PADDD XMM3, XMM7      ;sumamos estos pesos a los ya sumados previamente
	PSUBD XMM3, XMM5      ;sustraemos 128 por cada bit que estaba activo

	DEC ECX
	JNZ bucle3

	MOVD EAX, XMM3        ;copiamos la mitad baja del resultado
	PSRLDQ XMM3, 8        ;desplazamos el registro 8 bytes a la derecha
	MOVD ECX, XMM3        ;copiamos la mitad alta del resultado
	ADD EAX, ECX          ;sumamos ambas mitades (en EAX esta el resultado a devolver)

	;restauramos los registros
	POP EDI
	POP ESI
	POP ECX
	POP EBX

	RET


global XMMbipolar
XMMbipolar: 

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 20]  ;vector de entradas
	MOV ECX, [ESP + 24]  ;numero de entradas
	;SHR ECX, 4           ;dividimos entre 16 (el número de entradas que se procesan por vuelta del bucle)
	MOV EDI, [ESP + 28]  ;pesos

	MOVDQU XMM6, [mascara];iniciamos la mascara para cuando la tengamos que usar
	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él
	PXOR XMM0, XMM0       ;anulamos XMM0 para cuando lo tengamos que usar
	MOV BL, 1	      ;para que la primera vez se inicie la mascara 

bucle4:

	PSRLW XMM4, 1         ;desplazamos las mascaras de entradas
	DEC BL                ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara4 ;si BL==0, se terminó de procesar el bloque de entrada		

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV BL, 8             ;ciclos por bloque de entrada	
	MOVDQU XMM1, [ESI]    ;leemos el bloque de estados de entrada que toca
	ADD ESI, 16           ;actualizamos el puntero

noIniciarMascara4:

	MOVDQA XMM7, XMM4     ;copiamos la mascacara en una mascara auxiliar
	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM2, XMM2    ;ponemos a 1 todo el XMM2
	PXOR XMM2, XMM7	      ;en XMM2 esta el inverso de XMM7 


	MOVDQU XMM5, XMM6     ;128 en todo XMM5
	PAND XMM5, XMM2       ;128 en todos los bytes que estaban activos
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba activo
	PSUBD XMM3, XMM5      ;sustraemos 128 por cada bit que estaba activo
	
	MOVDQU XMM5, XMM6     ;128 en todo XMM5
	PAND XMM5, XMM7       ;128 en todos los bytes que estaban inactivos
	PSADBW XMM5, XMM0     ;sumamos 128 por cada byte que estaba inactivo
	PADDD XMM3, XMM5      ;sumamos 128 por cada bit que estaba inactivo


	MOVDQU XMM5, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos
	PAND XMM2, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit activo
	PAND XMM7, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit inactivo


	PSADBW XMM2, XMM0     ;sumamos todos los bytes (los que estaban activos)
	PSADBW XMM7, XMM0     ;sumamos todos los bytes (los que estaban inactivos)

	PADDD XMM3, XMM2      ;sumamos los pesos "positivos" al acumulador
	PSUBD XMM3, XMM7      ;sustraemos los pesos "negativos" al acumulador

	DEC ECX
	JNZ bucle4

	MOVD EAX, XMM3        ;copiamos la mitad baja del resultado
	PSRLDQ XMM3, 8        ;desplazamos el registro 8 bytes a la derecha
	MOVD ECX, XMM3        ;copiamos la mitad alta del resultado
	ADD EAX, ECX          ;sumamos ambas mitades (en EAX esta el resultado a devolver)

	;restauramos los registros
	POP EDI
	POP ESI
	POP ECX
	POP EBX

	RET


global XMMreal
XMMreal: ;PROC

	;guardamos el valor de los registros
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 16]  ;vector de entradas
	MOV ECX, [ESP + 20]  ;numero de entradas
	;SHR ECX, 2           ;dividimos entre 4 (el número de entradas que se procesan por vuelta del bucle)
	MOV EDI, [ESP + 24]  ;pesos

	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él

bucleReal:

	MOVUPS XMM0, [ESI]    ;leemos el bloque actual de entradas
	ADD ESI, 16           ;actualizamos el puntero de pesos
	MOVUPS XMM1, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos


	MULPS XMM0, XMM1      ;los multiplicamos
	ADDPS XMM3, XMM0      ;acumulamos los resultados

	DEC ECX
	JNZ bucleReal

	MOVAPS XMM4, XMM3
	SHUFPS XMM4, XMM4, 11101110b  ;ponemos los dos floats altos en la parte baja
	ADDPS XMM4, XMM3  ;sumamos los dos floats mas altos con los mas bajos
	MOVAPS XMM3, XMM4
	PSRLDQ XMM4, 4    ;desplazamos el registro 4 bytes a la derecha
	ADDPS XMM3, XMM4  ;sumamos la parte mas baja
	MOVD EAX, XMM3    ;copiamos el resultado

	MOV EDI, [ESP + 28]
	MOV [EDI], EAX        ;las colocamos en el parámetro resultado  

	;restauramos los registros
	POP EDI
	POP ESI
	POP ECX

	RET

global XMMreal2
XMMreal2: ;PROC

	;guardamos el valor de los registros
	PUSH EDX
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV EDX, [ESP + 24]  ;numIndicesEntradas
	MOV EDI, [ESP + 28]  ;pesos

	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él
	MOV EAX, 0

bucleVectorReal:

	MOV ESI, [ESP+20]    ;vectorIndicesEntradas
	ADD ESI, EAX

	MOV ECX, [ESI+4]     
	SHR ECX, 2           ;dividimos entre 4 (el número de entradas que se procesan por vuelta del bucle)

	MOV ESI, [ESI]

bucleReal2:

	MOVUPS XMM0, [ESI]    ;leemos el bloque actual de entradas
	ADD ESI, 16           ;actualizamos el puntero de pesos
	MOVUPS XMM1, [EDI]    ;leemos el bloque actual de pesos
	ADD EDI, 16           ;actualizamos el puntero de pesos


	MULPS XMM0, XMM1      ;los multiplicamos
	ADDPS XMM3, XMM0      ;acumulamos los resultados

	DEC ECX
	JNZ bucleReal2  

	ADD EAX, 8            ;le sumamos a EAX el tamaño de una estructura de las que hay en el vector
	DEC EDX
	JNZ bucleVectorReal

	MOVAPS XMM4, XMM3
	SHUFPS XMM4, XMM4, 11101110b  ;copiamos los dos floats altos en ambas partes
	ADDPS XMM4, XMM3  ;sumamos los dos floats mas altos con los mas bajos
	MOVAPS XMM5, XMM4
	PSRLDQ XMM4, 4    ;desplazamos el registro 4 bytes a la derecha
	ADDPS XMM5, XMM4  ;sumamos la parte mas baja
	MOVD EAX, XMM5    ;copiamos el resultado 

	MOV EDI, [ESP + 28]
	MOV [EDI], EAX        ;las colocamos en el parámetro resultado  

	;restauramos los registros
	POP EDI
	POP ESI
	POP ECX
	POP EDX

	RET

;END ;fin del fichero
