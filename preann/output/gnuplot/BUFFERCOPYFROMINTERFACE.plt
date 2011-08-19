set terminal png 
set output "/home/timon/workspace/preann/output/images/BUFFERCOPYFROMINTERFACE.png" 
plot      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:2 title "FLOAT_C" with linespoints lt 1 pt 2,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:3 title "FLOAT_SSE2" with linespoints lt 2 pt 2,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:4 title "FLOAT_CUDA" with linespoints lt 3 pt 2,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:5 title "FLOAT_CUDA2" with linespoints lt 5 pt 2,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:6 title "FLOAT_CUDA_INV" with linespoints lt -1 pt 2,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:7 title "BIT_C" with linespoints lt 1 pt 4,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:8 title "BIT_SSE2" with linespoints lt 2 pt 4,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:9 title "BIT_CUDA" with linespoints lt 3 pt 4,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:10 title "BIT_CUDA2" with linespoints lt 5 pt 4,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:11 title "BIT_CUDA_INV" with linespoints lt -1 pt 4,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:12 title "SIGN_C" with linespoints lt 1 pt 8,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:13 title "SIGN_SSE2" with linespoints lt 2 pt 8,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:14 title "SIGN_CUDA" with linespoints lt 3 pt 8,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:15 title "SIGN_CUDA2" with linespoints lt 5 pt 8,      "/home/timon/workspace/preann/output/data/BUFFERCOPYFROMINTERFACE.DAT" using 1:16 title "SIGN_CUDA_INV" with linespoints lt -1 pt 8
