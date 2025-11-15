set terminal png size 2048,1024 
set key below
set key box 
set output "/home/jt/sync/data_common/code/jt/preann/output/images/chronoGen_Selection.png" 
set title "chronoGen_Selection" 
set xlabel "population_NumSelection" 
set ylabel "Tiempo (ms)" 
plot  "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_ROULETTE_WHEEL_400.000000.DAT" using 1:2 title "ROULETTE_WHEEL_400.000000" with linespoints lt 1 pt 2 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_ROULETTE_WHEEL_500.000000.DAT" using 1:2 title "ROULETTE_WHEEL_500.000000" with linespoints lt 1 pt 8 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_RANKING_400.000000.DAT" using 1:2 title "RANKING_400.000000" with linespoints lt 2 pt 2 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_RANKING_500.000000.DAT" using 1:2 title "RANKING_500.000000" with linespoints lt 2 pt 8 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_5.000000_400.000000.DAT" using 1:2 title "TOURNAMENT_5.000000_400.000000" with linespoints lt 3 pt 2 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_5.000000_500.000000.DAT" using 1:2 title "TOURNAMENT_5.000000_500.000000" with linespoints lt 3 pt 8 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_15.000000_400.000000.DAT" using 1:2 title "TOURNAMENT_15.000000_400.000000" with linespoints lt 3 pt 4 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_15.000000_500.000000.DAT" using 1:2 title "TOURNAMENT_15.000000_500.000000" with linespoints lt 3 pt 12 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_25.000000_400.000000.DAT" using 1:2 title "TOURNAMENT_25.000000_400.000000" with linespoints lt 3 pt 10 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TOURNAMENT_25.000000_500.000000.DAT" using 1:2 title "TOURNAMENT_25.000000_500.000000" with linespoints lt 3 pt 6 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TRUNCATION_400.000000.DAT" using 1:2 title "TRUNCATION_400.000000" with linespoints lt 6 pt 2 , \
	 "/home/jt/sync/data_common/code/jt/preann/output/data/chronoGen_Selection_TRUNCATION_500.000000.DAT" using 1:2 title "TRUNCATION_500.000000" with linespoints lt 6 pt 8
