#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -check $1 -iter 5 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_32 -f ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 10 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -check $1 -iter 20 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -check $1 -iter 20 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -check $1 -iter 20 -partition 3 -rounds 20


#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -check $1 -iter 50 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_32 -f ../datasets/hollywood-2009/hollywood-2009_ud.csr -i 10 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_32_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_32_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_32_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -check $1 -iter 50 -rounds 20 -partition 3


#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_64_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -check $1 -iter 50 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_128 -f ../datasets/indochina-2004/indochina-2004_di.csr -i 10 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_64_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_64_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -check $1 -iter 50 -rounds 20 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_64_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -check $1 -iter 50 -rounds 20 -partition 3


#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_32_1 -file ../datasets/twitter/twitter.csr -check $1 -iter 100 -rounds 20
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_32 -f ../datasets/twitter/twitter.csr -i 200 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_32_1 -file ../datasets/twitter/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_32_1 -file ../datasets/twitter/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_32_1 -file ../datasets/twitter/twitter.csr -check $1 -partition 2 -iter 100 -rounds 20


#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_128_1 -file ../datasets/road_usa/road_usa_ud.csr -check $1 -iter 10 -rounds 10
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_128 -f ../datasets/road_usa/road_usa_ud.csr -i 10 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_512_1 -file ../datasets/road_usa/road_usa_ud.csr -check $1 -iter 40 -rounds 10 -partition 3
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_512_1 -file ../datasets/road_usa/road_usa_ud.csr -check $1 -iter 40 -rounds 10 -partition 3
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_512_1 -file ../datasets/road_usa/road_usa_ud.csr -check $1 -iter 40 -rounds 10 -partition 3

#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_128_1 -file ../datasets/osm-eur/osm-eur.csr -check $1 -iter 10 -rounds 10
/home/yuxin420/Atos/single-GPU/tests/pr_cta_discrete_128 -f ../datasets/osm-eur/osm-eur.csr -i 10 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_128_1 -file ../datasets/osm-eur/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_128_1 -file ../datasets/osm-eur/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_128_1 -file ../datasets/osm-eur/osm-eur.csr -partition 3 -check $1 -iter 50 -rounds 10
