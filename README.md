# ATOS

## Compile BFS
Under bfs\_nvlink directory, compile the code with:
```
make FETCHSIZE=32
make FETCHSIZE=64
make FETCHSIZE=128
make FETCHSIZE=256
```

To test the given datasets, run the figure5\_1.sh and figure5\_2.sh to generate performance data

## Compile PageRank
Under pr\_nvlink directory, compile the code with:
```
make FETCHSIZE=32
make FETCHSIZE=64
make FETCHSIZE=128
make FETCHSIZE=256
make FETCHSIZE=128 ROUND=2
make FETCHSIZE=256 ROUND=2
```

To test the given datasets, run the figure7\_1.sh and figure7\_2.sh to generate performance data

