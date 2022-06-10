if [ $# -eq 0 ]; then
  outfile="temp_pr_out.txt"
else
  outfile=$1
fi

echo "Extracting performance data from file:" $outfile

datasets=("soc-LiveJournal1" "hollywood_2009" "indochina_2004" "twitter" "road_usa" "osm-eur")
output=`awk '{if($1 == "ave" && $2 == "time:") print $3","}' $outfile`
#echo ${output}
outputtime=()
IFS=,
for i in $output
do
	outputtime+=($i)
done

idx=0
for graph in "${datasets[@]}"
do
	#echo ${graph}
	perf=()
	for GPU in 2 3 4 5 6 7 8
	do 
	    maxtime=0
	    #echo "GPU " ${GPU}
	    START=1
	    END=$GPU
	    for (( c=$START; c<=$END; c++ ))
	    do
		 #echo "max now: " $maxtime
		 #echo "pop time: " ${outputtime[$idx]}
		 temp=${outputtime[$idx]}
       	 compare=$(bc -l <<<"${temp}")
		 #echo -n "$compare "
         ifupdate=$(echo "$maxtime < $compare" | bc)
		 #echo "update? " $ifupdate
		 if [ $ifupdate -gt "0" ]; then
		    maxtime=$compare
		    #echo ${maxtime}
		 fi
		 idx=$(($idx + 1))
	    done
	    #echo "MAX: " $maxtime
        #echo " "
	    perf+=($maxtime)
	done
	echo ${graph}
	echo "2 GPUs      3 GPUs    4 GPUs      5 GPUs     6 GPUs      7 GPUs      8 GPUs"
	for i in "${perf[@]}"
	do
		echo -n "$i       "
	done
	echo " "

done

	

