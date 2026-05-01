; 64-bit (System V x86-64) port of src/sse2/sse2_sign.asm
; Algorithm and SSE2 instructions are preserved verbatim from the original.
; arg regs: RDI=bufferEntrada, ESI=numeroBloques, RDX=pesos
; result returned in EAX
; R10B is used in place of the original BL to avoid saving callee-saved RBX

; Mark stack as non-executable (modern Linux security hardening; NASM
; doesn't emit this section by default).
section .note.GNU-stack noalloc noexec nowrite progbits

section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

section .text

global XMMbipolar
XMMbipolar:

	MOVDQU XMM6, [rel mascara];iniciamos la mascara para cuando la tengamos que usar
	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él
	PXOR XMM0, XMM0       ;anulamos XMM0 para cuando lo tengamos que usar
	MOV R10B, 1	          ;para que la primera vez se inicie la mascara

bucle4:

	PSRLW XMM4, 1         ;desplazamos las mascaras de entradas
	DEC R10B              ;decrementamos el contador de las mascaras de entradas
	JNZ noIniciarMascara4 ;si R10B==0, se terminó de procesar el bloque de entrada

	MOVDQA XMM4, XMM6     ;reiniciamos la mascara
	MOV R10B, 8           ;ciclos por bloque de entrada
	MOVDQU XMM1, [RDI]    ;leemos el bloque de estados de entrada que toca
	ADD RDI, 16           ;actualizamos el puntero

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


	MOVDQU XMM5, [RDX]    ;leemos el bloque actual de pesos
	ADD RDX, 16           ;actualizamos el puntero de pesos
	PAND XMM2, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit activo
	PAND XMM7, XMM5       ;asi tenemos el peso de cada conexión solamente en los bytes con el bit inactivo


	PSADBW XMM2, XMM0     ;sumamos todos los bytes (los que estaban activos)
	PSADBW XMM7, XMM0     ;sumamos todos los bytes (los que estaban inactivos)

	PADDD XMM3, XMM2      ;sumamos los pesos "positivos" al acumulador
	PSUBD XMM3, XMM7      ;sustraemos los pesos "negativos" al acumulador

	DEC ESI
	JNZ bucle4

	MOVD EAX, XMM3        ;copiamos la mitad baja del resultado
	PSRLDQ XMM3, 8        ;desplazamos el registro 8 bytes a la derecha
	MOVD ECX, XMM3        ;copiamos la mitad alta del resultado
	ADD EAX, ECX          ;sumamos ambas mitades (en EAX esta el resultado a devolver)

	RET
