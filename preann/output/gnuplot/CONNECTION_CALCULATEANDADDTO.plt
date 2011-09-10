set terminal png 
set output "/home/timon/workspace/preann/output/images/CONNECTION_CALCULATEANDADDTO.png" 
plot      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:2 title "FLOAT_C" with linespoints lt 1 pt 2,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:3 title "FLOAT_SSE2" with linespoints lt 2 pt 2,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:4 title "FLOAT_CUDA_REDUC" with linespoints lt 5 pt 2,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:5 title "FLOAT_CUDA_INV" with linespoints lt -1 pt 2,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:6 title "BIT_C" with linespoints lt 1 pt 4,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:7 title "BIT_SSE2" with linespoints lt 2 pt 4,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:8 title "BIT_CUDA_REDUC" with linespoints lt 5 pt 4,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:9 title "BIT_CUDA_INV" with linespoints lt -1 pt 4,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:10 title "SIGN_C" with linespoints lt 1 pt 8,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:11 title "SIGN_SSE2" with linespoints lt 2 pt 8,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:12 title "SIGN_CUDA_REDUC" with linespoints lt 5 pt 8,      "/home/timon/workspace/preann/output/data/CONNECTION_CALCULATEANDADDTO.DAT" using 1:13 title "SIGN_CUDA_INV" with linespoints lt -1 pt 8
