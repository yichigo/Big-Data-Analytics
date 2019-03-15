
Step 1:
in:
code/hive/
run:
hive -f event_statistics.hql

Step 2:
in:
code/pig/
run:
sudo pig -x local etl.pig

Step 3:
in:
code/lr/
run:
1.
cat ../pig/training/part-r-00000 | python train.py -f 3618 -e 0.1 -c 0.001
2.
cat ../pig/testing/part-r-00000 | python test.py

Repeat to tune the parameters, above might be best choice


Step 4:
in:
code/
run:
sudo su - hdfs
hdfs dfs -mkdir /hw2
hdfs dfs -chown -R root /hw2
exit
hdfs dfs -put pig/training /hw2

Step 5:
in
code/
run:
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -D mapreduce.job.reduces=5 -files lr -mapper "python lr/mapper.py -n 5 -r 0.4" -reducer "python lr/reducer.py -e 0.4 -c 0.001 -f 3618" -input /hw2/training -output /hw2/models



Get data from hdfs:
hdfs dfs -get /hw2/models

Test the ensemble ROC:
cat pig/testing/* | python lr/testensemble.py -m models


Step 5.repeat:
run:
sudo su - hdfs
hdfs dfs -rm -r /hw2/models
exit

in
code/
run:
hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -D mapreduce.job.reduces=5 -files lr -mapper "python lr/mapper.py -n 5 -r 0.4" -reducer "python lr/reducer.py -e 0.4 -c 0.001 -f 3618" -input /hw2/training -output /hw2/models

rm -rf models
hdfs dfs -get /hw2/models
cat pig/testing/* | python lr/testensemble.py -m models


