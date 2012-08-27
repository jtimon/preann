section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

global XMMbipolar
XMMbipolar:

	;guardamos el valor de los registros
	PUSH EBX
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 20]  ;buffer de entradas
	MOV ECX, [ESP + 24]  ;numero de entradas
	;SHR ECX, 4           ;dividimos entre 16 (el número de entradas que se procesan por vuelta del bucle)
	MOV EDI, [ESP + 28]  ;pesos

	MOVDQU XMM6, [mascara];iniciamos la mascara para cuando la tengamos que usar
	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él
	PXOR XMM0, XMM0       ;anulamos XMM0 para cuando lo tengamos que usar
	MOV BL, 1	          ;para que la primera vez se inicie la mascara

bucle4:

	PSRLW XMM4, 1         ;desplazamos las mascaras de entradas
	DEC BL                ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara4 ;si BL==0, se terminó de procesar el bloque de entrada

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV BL, 8             ;ciclos por bloque de entrada
	MOVDQU XMM1, [ESI]    ;leemos el bloque de estados de entrada que toca
	ADD ESI, 16           ;actualizamos el puntero

noIniciarMascara4:

	MOVDQA XMM7, XMM4     ;copiamos la máscara en un registro auxiliar
	PAND XMM7, XMM1       ;obtenemos el valor del bit a procesar en cada byte
	PCMPEQB XMM7, XMM0    ;si el bit estaba activo->se pone a 0 todo el byte, si no-> se pone a 1 todo el byte
	PCMPEQB XMM2, XMM2    ;ponemos 255 en todos los byes del registro XXM2
	PXOR XMM2, XMM7	      ;en XMM2 ponemos el inverso de XMM7

	MOVDQU XMM5, XMM6     ;128 en todo XMM5
	PAND XMM5, XMM2       ;128 sólo en los bytes que estaban activos
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
