$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr  -v 1 -check $1 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr  -v 1 -check $1 -iter 150 -ratio 0.8 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr  -v 1 -check $1 -iter 150 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/soc-LiveJournal1.csr  -v 1 -check $1 -iter 150 -ratio 0.8 -partition 3 -rounds 20



$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -check $1 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -check $1 -iter 100 -ratio 2.5 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -check $1 -iter 200 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/hollywood-2009.csr -v 1 -check $1 -iter 150 -ratio 1.2 -partition 3 -rounds 10




$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -chek 1 -iter 400 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -check $1 -iter 500 -ratio 0.6 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -check $1 -iter 400 -ratio 0.7 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_1 -file /data/yuxin/3_graph_dataset/indochina-2004.csr -v 1 -check $1 -iter 400 -ratio 0.6 -partition 3 -rounds 20




$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -check $1 -rounds 10 -iter 900
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -check $1 -partition 2 -rounds 10 -iter 900
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -check $1 -partition 2 -rounds 10 -iter 900
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_64_1 -file /data/yuxin/3_graph_dataset/twitter.csr -v 1 -check $1 -iter 800 -partition 2 -rounds 20





$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_256_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -check $1 -iter 400 -rounds 10
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_256_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -check $1 -iter 200 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_256_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -check $1 -iter 150 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_256_1 -file /data/yuxin/3_graph_dataset/road_usa.csr -v 1 -check $1 -iter 100 -ratio 1 -partition 3 -rounds 10





$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_256_2 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -check $1 -iter 6500 -rounds 10
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_2 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -partition 3 -check $1 -iter 3000 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_2 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -partition 3 -check $1 -iter 3000 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_2 -file /data/yuxin/3_graph_dataset/osm-eur.csr -v 1 -check $1 -iter 3000 -partition 3 -rounds 10
