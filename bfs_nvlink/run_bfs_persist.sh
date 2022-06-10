$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -iter 100 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -partition 3 -iter 100 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -partition 3 -iter 200 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -partition 3 -iter 300 -check $1 -rounds 20


$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -iter 100 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -partition 3 -iter 600 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -partition 3 -iter 700 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -partition 0 -iter 1000 -check $1 -rounds 20



$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -iter 100 -check $1 -rounds 20 -source 40
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -partition 3 -iter 600 -check $1 -rounds 20 -source 40
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -partition 3 -iter 800 -check $1 -rounds 20 -source 40
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -partition 3 -iter 800 -check $1 -rounds 20 -source 40



$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/twitter.csr -iter 100 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/twitter.csr -partition 2 -iter 800 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/twitter.csr -partition 2 -iter 800 -check $1 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/twitter.csr -partition 2 -iter 1200 -check $1 -rounds 20


$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -iter 5000 -rounds 10 -check $1
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -partition 3 -iter 10000 -rounds 10 -check $1 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -partition 3 -iter 11000 -rounds 10 -check $1
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -partition 3 -iter 12000 -ratio 1.3 -rounds 10 -check $1





$MPI_HOME/bin/mpirun -n 1 ./test_bfs_queue_persist_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -check $1 -iter 30000 -rounds 10 
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_queue_persist_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -partition 3 -check $1 -iter 45000 -ration 0.9 -rounds 10 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_queue_persist_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -partition 3 -check $1 -iter 50000 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_queue_persist_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -partition 3 -check $1 -iter 55000 -rounds 10
