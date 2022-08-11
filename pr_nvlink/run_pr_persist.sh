#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr  -v 1 -check $1 -iter 100 -rounds 20
../single-GPU/tests/pr_cta_32 -f ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 150 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr  -v 1 -check $1 -iter 150 -ratio 0.8 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr  -v 1 -check $1 -iter 150 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_64_1 -file ../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr  -v 1 -check $1 -iter 150 -ratio 0.8 -partition 3 -rounds 20



#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_128_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -v 1 -check $1 -iter 100 -rounds 20
../single-GPU/tests/pr_cta_128 -f ../datasets/hollywood-2009/hollywood-2009_ud.csr -i 200 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -v 1 -check $1 -iter 100 -ratio 2.5 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -v 1 -check $1 -iter 200 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_1 -file ../datasets/hollywood-2009/hollywood-2009_ud.csr -v 1 -check $1 -iter 150 -ratio 1.2 -partition 3 -rounds 10




#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_128_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -v 1 -chek 1 -iter 400 -rounds 20
../single-GPU/tests/pr_cta_128 -f ../datasets/indochina-2004/indochina-2004_di.csr -i 750 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -v 1 -check $1 -iter 500 -ratio 0.6 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -v 1 -check $1 -iter 400 -ratio 0.7 -partition 3 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_1 -file ../datasets/indochina-2004/indochina-2004_di.csr -v 1 -check $1 -iter 400 -ratio 0.6 -partition 3 -rounds 20




#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_64_1 -file ../datasets/twitter/twitter.csr -v 1 -check $1 -rounds 10 -iter 900
../single-GPU/tests/pr_cta_256_10 -f ../datasets/twitter/twitter.csr -i 200 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_64_1 -file ../datasets/twitter/twitter.csr -v 1 -check $1 -partition 2 -rounds 10 -iter 900
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_64_1 -file ../datasets/twitter/twitter.csr -v 1 -check $1 -partition 2 -rounds 10 -iter 900
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_64_1 -file ../datasets/twitter/twitter.csr -v 1 -check $1 -iter 800 -partition 2 -rounds 20





#$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_256_1 -file ../datasets/road_usa/road_usa_ud.csr -v 1 -check $1 -iter 400 -rounds 10
../single-GPU/tests/pr_cta_256 -f ../datasets/road_usa/road_usa_ud.csr -i 400 -rounds 20
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_256_1 -file ../datasets/road_usa/road_usa_ud.csr -v 1 -check $1 -iter 200 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_256_1 -file ../datasets/road_usa/road_usa_ud.csr -v 1 -check $1 -iter 150 -ratio 1 -partition 3 -rounds 10
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_256_1 -file ../datasets/road_usa/road_usa_ud.csr -v 1 -check $1 -iter 100 -ratio 1 -partition 3 -rounds 10





$MPI_HOME/bin/mpirun -n 1 ./test_pr_queue_persist_256_2 -file ../datasets/osm-eur/osm-eur.csr -v 1 -check $1 -iter 6500 -rounds 10
$MPI_HOME/bin/mpirun -n 2 ./test_pr_queue_persist_128_2 -file ../datasets/osm-eur/osm-eur.csr -v 1 -partition 3 -check $1 -iter 3000 -rounds 20
$MPI_HOME/bin/mpirun -n 3 ./test_pr_queue_persist_128_2 -file ../datasets/osm-eur/osm-eur.csr -v 1 -partition 3 -check $1 -iter 3000 -rounds 20
$MPI_HOME/bin/mpirun -n 4 ./test_pr_queue_persist_128_2 -file ../datasets/osm-eur/osm-eur.csr -v 1 -check $1 -iter 3000 -partition 3 -rounds 10
