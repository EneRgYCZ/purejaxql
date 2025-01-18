for in in `seq 1 11`
do
     python purejaxql/transfer_learning.py +alg=pqn_minatar_transfer
     echo "Run $in completed"
     sleep 20
done