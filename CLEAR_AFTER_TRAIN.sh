date=$1
best_epoch=$2
lambda=$3

json_file="result_${lambda}.json"
train_result_file="result_${lambda}.txt"
cal_result_file="result_${lambda}_cal.txt"

mkdir -p stats/stats_${date}/

stats_dir=stats_${date}/${lambda}_nws/
results_dir=result_$date/
models_dir=models/model_$date/${lambda}_nws/

rm -rf $results_dir

for i in $(find -L $stats_dir -mindepth 1 -maxdepth 1 | sort); do
    filename=$(basename $i)
    realname=${filename%.*}
    epoch=$(echo $realname | awk '{split($0,tmp,"_"); print tmp[4]}')
    if [ $epoch -eq $best_epoch ]; then
        continue
    elif [ $epoch -eq '2000' ]; then
        continue
    elif [ $epoch -eq '4000' ]; then
        continue
    else
        rm -rf $i
    fi
done

mv $json_file $stats_dir
mv $train_result_file $stats_dir
mv $cal_result_file $stats_dir
mv $stats_dir stats/stats_$date

for i in $(find -L $models_dir -mindepth 1 -maxdepth 1 | sort); do
    filename=$(basename $i)
    epoch=$(echo $filename | awk '{split($0,tmp,"_"); print[1]}')
    if [ $epoch -eq $best_epoch ]; then
        continue
    elif [ $epoch -eq '4000' ]; then
        continue
    else
       rm -rf $i
    fi
done
