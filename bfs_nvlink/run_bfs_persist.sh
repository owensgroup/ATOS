#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_64 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -iter 100 -check $1 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/bfs_32 -f ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -o 3 -q 1 -i 50 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_64 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -partition 3 -iter 100 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_64 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -partition 3 -iter 200 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_64 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -partition 3 -iter 300 -check $1 -rounds 20


#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -iter 100 -check $1 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/bfs_32 -f ../datasets/hollywood-2009/hollywood-2009_ud.csr -o 3 -q 1 -i 50 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -partition 3 -iter 600 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -partition 3 -iter 700 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -partition 0 -iter 1000 -check $1 -rounds 20



#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_64 -file ../datasets/indochina-2004/indochina-2004_di.csr -iter 100 -check $1 -rounds 20 -source 40
/home/yuxin420/Atos/single-GPU/tests/bfs_128 -f ../datasets/indochina-2004/indochina-2004_di.csr -o 3 -q 1 -i 70 -r 40 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_64 -file ../datasets/indochina-2004/indochina-2004_di.csr -partition 3 -iter 600 -check $1 -rounds 20 -source 40
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_64 -file ../datasets/indochina-2004/indochina-2004_di.csr -partition 3 -iter 800 -check $1 -rounds 20 -source 40
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_64 -file ../datasets/indochina-2004/indochina-2004_di.csr -partition 3 -iter 800 -check $1 -rounds 20 -source 40



#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file ../datasets/twitter/twitter.csr -iter 100 -check $1 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/bfs_32 -f ../datasets/twitter/twitter.csr -o 3 -q 1 -i 100 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file ../datasets/twitter/twitter.csr -partition 2 -iter 800 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file ../datasets/twitter/twitter.csr -partition 2 -iter 800 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file ../datasets/twitter/twitter.csr -partition 2 -iter 1200 -check $1 -rounds 20


#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file ../datasets/road_usa/road_usa_ud.csr -iter 5000 -rounds 10 -check $1
/home/yuxin420/Atos/single-GPU/tests/bfs_32 -f ../datasets/road_usa/road_usa_ud.csr -o 3 -q 1 -i 7000 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file ../datasets/road_usa/road_usa_ud.csr -partition 3 -iter 10000 -rounds 10 -check $1 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file ../datasets/road_usa/road_usa_ud.csr -partition 3 -iter 11000 -rounds 10 -check $1
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file ../datasets/road_usa/road_usa_ud.csr -partition 3 -iter 12000 -ratio 1.3 -rounds 10 -check $1





#$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_128 -file ../datasets/osm-eur/osm-eur.csr -v 1 -check $1 -iter 30000 -rounds 10 
/home/yuxin420/Atos/single-GPU/tests/bfs_128 -f ../datasets/osm-eur/osm-eur.csr -o 3 -q 1 -i 51000 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_128 -file ../datasets/osm-eur/osm-eur.csr -v 1 -partition 3 -check $1 -iter 45000 -ration 0.9 -rounds 10 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_128 -file ../datasets/osm-eur/osm-eur.csr -v 1 -partition 3 -check $1 -iter 50000 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_128 -file ../datasets/osm-eur/osm-eur.csr -v 1 -partition 3 -check $1 -iter 55000 -rounds 10
