set terminal png 
set output "/home/timon/workspace/preann/output/images/CONNECTION_MUTATE.png" 
plot  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEFLOAT_C.DAT" using 1:2 title "FLOAT_C" with linespoints lt 1 pt 2 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEFLOAT_SSE2.DAT" using 1:2 title "FLOAT_SSE2" with linespoints lt 2 pt 2 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEFLOAT_CUDA.DAT" using 1:2 title "FLOAT_CUDA" with linespoints lt 3 pt 2 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEFLOAT_CUDA_REDUC.DAT" using 1:2 title "FLOAT_CUDA_REDUC" with linespoints lt 5 pt 2 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEFLOAT_CUDA_INV.DAT" using 1:2 title "FLOAT_CUDA_INV" with linespoints lt -1 pt 2 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEBIT_C.DAT" using 1:2 title "BIT_C" with linespoints lt 1 pt 6 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEBIT_SSE2.DAT" using 1:2 title "BIT_SSE2" with linespoints lt 2 pt 6 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEBIT_CUDA.DAT" using 1:2 title "BIT_CUDA" with linespoints lt 3 pt 6 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEBIT_CUDA_REDUC.DAT" using 1:2 title "BIT_CUDA_REDUC" with linespoints lt 5 pt 6 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATEBIT_CUDA_INV.DAT" using 1:2 title "BIT_CUDA_INV" with linespoints lt -1 pt 6 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATESIGN_C.DAT" using 1:2 title "SIGN_C" with linespoints lt 1 pt 4 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATESIGN_SSE2.DAT" using 1:2 title "SIGN_SSE2" with linespoints lt 2 pt 4 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATESIGN_CUDA.DAT" using 1:2 title "SIGN_CUDA" with linespoints lt 3 pt 4 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATESIGN_CUDA_REDUC.DAT" using 1:2 title "SIGN_CUDA_REDUC" with linespoints lt 5 pt 4 ,  "/home/timon/workspace/preann/output/data/CONNECTION_MUTATESIGN_CUDA_INV.DAT" using 1:2 title "SIGN_CUDA_INV" with linespoints lt -1 pt 4
