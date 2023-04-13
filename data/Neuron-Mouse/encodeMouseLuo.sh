count=0
sp="/-\|"
for cell in $(ls -d *indexed)
do
    count=$((count+1))
    echo Encoding: $count
    python ../EncodeLabelsLuo.py $cell X_Luo_Mouse.npz y_"$count".npz pos_"$count".npz --prepend_chr --chroms 10 11 12 13 14 15 16 17 18 19 1 2 3 4 5 6 7 8 9 X Y
done