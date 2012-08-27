section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

global XMMreal
XMMreal: ;PROC

	;guardamos el valor de los registros
	PUSH ECX
	PUSH ESI
	PUSH EDI

	MOV ESI, [ESP + 16]  ;buffer de entradas
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
