df

CONFIG=$1

# bash tools/my_dist_train.sh configs/$CONFIG.py 8
bash tools/my_dist_train2.sh configs/$CONFIG.py 8
# bash tools/my_dist_train3.sh configs/$CONFIG.py 2