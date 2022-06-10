$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -check $1 -iter 5 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -check $1 -iter 20 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -check $1 -iter 20 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr -check $1 -iter 20 -partition 3 -rounds 20


$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -check $1 -iter 50 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -check $1 -iter 50 -rounds 20 -partition 3


$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -check $1 -iter 50 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_64_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -check $1 -iter 50 -rounds 20 -partition 3


$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/twitter.csr -check $1 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_32_1 -file /data/yuxin/3_graph_dataset/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20


$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_128_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -check $1 -iter 10 -rounds 10
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_512_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -check $1 -iter 40 -rounds 10 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_512_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -check $1 -iter 40 -rounds 10 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_512_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -check $1 -iter 40 -rounds 10 -partition 3

$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_128_1 -file /data/yuxin/3_graph_dataset/osm-eur.csr -check $1 -iter 10 -rounds 10
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_128_1 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_128_1 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_128_1 -file /data/yuxin/3_graph_dataset/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
