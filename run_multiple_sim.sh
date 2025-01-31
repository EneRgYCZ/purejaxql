for in in `seq 1 5`
do
     python purejaxql/transfer_learning.py +alg=pqn_minatar_transfer_2
     echo "Run $in completed"
     sleep 20
done
for in in `seq 1 10`
do
     python purejaxql/transfer_learning.py +alg=pqn_minatar_transfer_3
     echo "Run $in completed"
     sleep 20
done