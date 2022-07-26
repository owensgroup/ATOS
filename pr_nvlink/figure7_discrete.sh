make all -C ../single-GPU/tests/
echo "Do you want to run this script with quick mode?"
read -r ifquick

modequick=0
outfile="temp_pr_quick_out.txt"
if [ $ifquick == "yes" ] || [ $ifquick == "Yes" ] || [ $ifquick == "YES" ] || [ $ifquick == "1" ]; then
        echo "run with quick mode"
	modequick=0
else
        echo "run with check mode"
	modequick=1
	outfile="temp_pr_check_out.txt"
fi

if test -f "$outfile"; then
	echo "Do you want to regenerate performance data?"
	read -r ifgenerate
	if [ $ifgenerate == "yes" ] || [ $ifgenerate == "Yes" ] || [ $ifgenerate == "YES" ] || [ $ifgenerate == "1" ]; then
    	echo "Remove $outfile"
    	rm $outfile
	fi
fi

if ! test -f "$outfile"; then 
	echo "Generating performance data"
	./run_pr.sh $modequick >> $outfile
fi

datasets=("soc-LiveJournal1" "hollywood_2009" "indochina_2004" "twitter" "road_usa" "osm-eur")
output=`awk '{if($1 == "ave" && $2 == "time:") print $3","; if($1 == "Ave." && $2 == "Time:") print $3","}' $outfile`
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
	for GPU in 1 2 3 4
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
	echo "1 GPU     2 GPUs      3 GPUs    4 GPUs"
	for i in "${perf[@]}"
	do
		echo -n "$i       "
	done
	echo " "

done
