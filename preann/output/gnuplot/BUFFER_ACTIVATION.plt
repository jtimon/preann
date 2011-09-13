set terminal png 
set output "/home/timon/workspace/preann/output/images/BUFFER_ACTIVATION.png" 
plot  "/home/timon/workspace/preann/output/data/BUFFER_ACTIVATION.DAT" using 1:2 title "" with linespoints lt 1 pt 2
