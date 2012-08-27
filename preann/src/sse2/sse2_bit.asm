section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

;extern "C" int XMMbinario (void* bufferEntrada, unsigned numeroBloques, unsigned char* pesos);
global XMMbinario
XMMbinario:

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 20]  ;buffer de entradas
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

	MOVDQA XMM7, XMM4     ;copiamos la máscara en un registro auxiliar

	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM5, XMM5    ;ponemos 255 en todos los byes del registro XXM5
	PXOR XMM7, XMM5	      ;invertimos XMM7 (ahora hay 255 en los bytes que tenian el bit que tocaba activo)

	MOVDQU XMM5, XMM6     ;128 en todos los bytes de XMM5
	PAND XMM5, XMM7       ;128 sólo en los bytes que estaban activos

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
