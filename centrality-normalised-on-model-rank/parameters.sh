for c in 1 2 3 4
do
  for d in 1 2 4 8 16 32 64
  do
    echo "$c" "$d" $(python3 model.py $c $d)
  done
done
