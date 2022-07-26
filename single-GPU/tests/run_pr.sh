#round 1
#fetch size 256
echo "./pr -f ../../datasets/road_usa/road_usa_ud.csr -rounds 10 -i 400 -o 2"
./pr -f ../../datasets/road_usa/road_usa_ud.csr -rounds 10 -i 400

echo "./pr_cta_256 -f ../../datasets/road_usa/road_usa_ud.csr -i 400 -rounds 20"
./pr_cta_256 -f ../../datasets/road_usa/road_usa_ud.csr -i 400 -rounds 20

echo "./pr_cta_discrete_128 -f ../../datasets/road_usa/road_usa_ud.csr -i 10 -rounds 20"
./pr_cta_discrete_128 -f ../../datasets/road_usa/road_usa_ud.csr -i 10 -rounds 20

echo "./pr -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -rounds 10 -i 100 -o 2"
./pr -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -rounds 10 -i 100

echo "./pr_cta_256 -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -i 60 -rounds 20"
./pr_cta_256 -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -i 60 -rounds 20

echo "./pr_cta_discrete_256 -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -i 10 -rounds 20"
./pr_cta_discrete_256 -f ../../datasets/roadNet-CA/roadNet-CA_ud.csr -i 10 -rounds 20


#fetch size 128
echo "./pr -f ../../datasets/indochina-2004/indochina-2004_di.csr -rounds 10 -i 1600 -o 2"
./pr -f ../../datasets/indochina-2004/indochina-2004_di.csr -rounds 10 -i 1600

echo "./pr_cta_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 750 -rounds 20"
./pr_cta_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 750 -rounds 20

echo "/pr_cta_discrete_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 10 -rounds 20"
./pr_cta_discrete_128 -f ../../datasets/indochina-2004/indochina-2004_di.csr -i 10 -rounds 20

echo "./pr -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -rounds 10 -i 500 -o 2"
./pr -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -rounds 10 -i 500

echo "./pr_cta_128 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 200 -rounds 20"
./pr_cta_128 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 200 -rounds 20

echo "./pr_cta_discrete_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 10 -rounds 20"
./pr_cta_discrete_32 -f ../../datasets/hollywood-2009/hollywood-2009_ud.csr -i 10 -rounds 20

#fetch size 32

echo "./pr -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -rounds 10 -i 200 -o 2"
./pr -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -rounds 10 -i 200

echo "./pr_cta_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 150 -rounds 20"
./pr_cta_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 150 -rounds 20

echo "./pr_cta_discrete_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 10 -rounds 20"
./pr_cta_discrete_32 -f ../../datasets/soc-LiveJournal1/soc-LiveJournal1_di.csr -i 10 -rounds 20

#fetch size 256 round 20

#echo "./pr_cta_256_20 -f ../../datasets/osm-eur/osm-eur.csr -i 1400 -rounds 20 -check 1"
#./pr_cta_256_20 -f ../../datasets/osm-eur/osm-eur.csr -i 1400 -rounds 20 -check 1 -o 2

echo "./pr_cta_discrete_128 -f ../../datasets/osm-eur/osm-eur.csr -i 10 -rounds 20"
./pr_cta_discrete_128 -f ../../datasets/osm-eur/osm-eur.csr -i 10 -rounds 20

##fetch size 256 round 10
echo "./pr_cta_256_10 -f ../../datasets/twitter/twitter.csr -i 200 -rounds 20"
./pr_cta_256_10 -f ../../datasets/twitter/twitter.csr -i 200 -rounds 20

echo "./pr_cta_discrete_32 -f ../../datasets/twitter/twitter.csr -i 200 -rounds 20"
./pr_cta_discrete_32 -f ../../datasets/twitter/twitter.csr -i 200 -rounds 20
