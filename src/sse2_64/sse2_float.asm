; 64-bit (System V x86-64) port of src/sse2/sse2_float.asm
; Algorithm and SSE2 instructions are preserved verbatim from the original.
; Only the calling convention and integer registers used as pointers change:
;   args: RDI=bufferEntrada, ESI=numeroBloques, RDX=pesos, RCX=&resultado
;   data is accessed RIP-relative ([rel mascara])

; Mark stack as non-executable (modern Linux security hardening; NASM
; doesn't emit this section by default).
section .note.GNU-stack noalloc noexec nowrite progbits

section .data

mascara:    db 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128

section .text

global XMMreal
XMMreal: ;PROC

	;arg regs: RDI=entradas, ESI=contador, RDX=pesos, RCX=puntero resultado

	PXOR XMM3, XMM3       ;anulamos XMM3 para acumular los sumatorios en él

bucleReal:

	MOVUPS XMM0, [RDI]    ;leemos el bloque actual de entradas
	ADD RDI, 16           ;actualizamos el puntero de entradas
	MOVUPS XMM1, [RDX]    ;leemos el bloque actual de pesos
	ADD RDX, 16           ;actualizamos el puntero de pesos


	MULPS XMM0, XMM1      ;los multiplicamos
	ADDPS XMM3, XMM0      ;acumulamos los resultados

	DEC ESI
	JNZ bucleReal

	MOVAPS XMM4, XMM3
	SHUFPS XMM4, XMM4, 11101110b  ;ponemos los dos floats altos en la parte baja
	ADDPS XMM4, XMM3  ;sumamos los dos floats mas altos con los mas bajos
	MOVAPS XMM3, XMM4
	PSRLDQ XMM4, 4    ;desplazamos el registro 4 bytes a la derecha
	ADDPS XMM3, XMM4  ;sumamos la parte mas baja
	MOVD EAX, XMM3    ;copiamos el resultado

	MOV [RCX], EAX        ;las colocamos en el parámetro resultado

	RET
