./gc_32 -run_bsp_async 2 -o 3 -f ~/temp/large_twitch_ud.csr -rounds 20 -i 300
./gc_32 -run_bsp_async 3 -f ~/temp/large_twitch_ud.csr -rounds 20 -i 50
./gc_32 -run_bsp_async 2 -o 2 -f ~/temp/large_twitch_ud.csr -rounds 20 -i 1500

./gc_32 -run_bsp_async 2 -o 2 -f ~/temp/cage13_ud.csr -rounds 20 -i 20
./gc_32 -run_bsp_async 2 -o 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 10
./gc_32 -run_bsp_async 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 10

./gc_128 -run_bsp_async 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 10
./gc_128 -run_bsp_async 2 -o 2 -f ~/temp/cage13_ud.csr -rounds 20 -i 10
./gc_128 -run_bsp_async 2 -o 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 10

./gc_256 -run_bsp_async 2 -o 2 -f ~/temp/cage13_ud.csr -rounds 20 -i 30
./gc_256 -run_bsp_async 2 -o 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 30
./gc_256 -run_bsp_async 3 -f ~/temp/cage13_ud.csr -rounds 20 -i 10

./gc_32 -run_bsp_async 2 -o 2 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 20
./gc_32 -run_bsp_async 2 -o 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10
./gc_32 -run_bsp_async 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10

./gc_128 -run_bsp_async 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10
./gc_128 -run_bsp_async 2 -o 2 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 20
./gc_128 -run_bsp_async 2 -o 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10

./gc_256 -run_bsp_async 2 -o 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10
./gc_256 -run_bsp_async 2 -o 2 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10
./gc_256 -run_bsp_async 3 -f ~/temp/atmosmodd_ud.csr -rounds 20 -i 10
