count=0
for cell in $(ls -d *bismark)
do
    count=$((count+1))
    # python ../EncodeLabelsLuo.py $cell X_HCC.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 1 20 21 22 2 3 4 5 6 7 8 9 X Y 
    # echo Encoding: $count   
    if test "$count" = "211"
    then
        echo "Process:" $count $cell
        python ../EncodeLabelsLuo.py $cell X_HCC.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 1 20 21 22 2 3 4 5 6 7 8 9 X Y 
    fi
done