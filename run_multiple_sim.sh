for in in `seq 1 20`
do
     python purejaxql/pqn_minatar.py +alg=pqn_minatar
     echo "Run $in completed"
     sleep 20
done