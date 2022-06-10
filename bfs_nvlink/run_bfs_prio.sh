$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -v 1 -check $1 -threshold 4 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -v 1 -partition 3 -check $1 -threshold 4 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -v 1 -partition 3 -check $1 -threshold 4 -rounds 20 
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -v 1 -partition 3 -check $1 -threshold 4 -rounds 20 



$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -check $1 -threshold 4 -rounds 20 
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -partition 3 -check $1 -ratio 2 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -partition 3 -iter 40 -check $1 -rounds 20 
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -partition 3 -iter 20 -check $1 -ratio 2 -rounds 20 




$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -check $1 -source 40 -rounds 20 
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -partition 3 -check $1 -threshold 4 -source 40 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -partition 3 -check $1 -threshold 4 -source 40 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_64 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -partition 3 -check $1 -threshold 4 -source 40 -rounds 20




$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -check $1 -rounds 20 
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -partition 2 -check $1 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -partition 2 -check $1 -rounds 20 
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -partition 2 -check $1 -rounds 20 




$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -check $1 -threshold 100 -delta 100 -rounds 20 
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -partition 3 -iter 500 -ratio 2 -check $1 -threshold 100 -delta 100 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -partition 3 -iter 800 -ratio 2 -check $1 -threshold 100 -delta 100 -rounds 20 
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_32 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -partition 3 -iter 1100 -ratio 2 -check $1 -threshold 100 -delta 100 -rounds 10 





$MPI_HOME/bin/mpirun -n 1 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -check $1 -threshold 80000 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -v 1 -check $1 -threshold 80000 -iter 1500 -rounds 20 
$MPI_HOME/bin/mpirun -n 3 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -v 1 -check $1 -threshold 80000 -iter 2000 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_bfs_priority_queue_128 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -v 1 -check $1 -threshold 80000 -iter 5000 -rounds 20
