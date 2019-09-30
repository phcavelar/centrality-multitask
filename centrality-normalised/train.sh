start=$(date +"%T")
echo "Start: $start"
#now=$(date +"%T")
#echo "B: $now"
#python3 training.py betweenness 1>1b.out 2>1b.err
#now=$(date +"%T")
#echo "C: $now"
#python3 training.py closeness 1>1c.out 2>1c.err
now=$(date +"%T")
echo "D: $now"
python3 training.py degree 1>1d.out 2>1d.err
#now=$(date +"%T")
#echo "E: $now"
#python3 training.py eigenvector 1>1e.out 2>1e.err
now=$(date +"%T")
echo "BCDE: $now"
python3 training.py betweenness closeness degree eigenvector 1>4bcde.out 2>4bcde.err
end=$(date +"%T")
echo "End: $end"
