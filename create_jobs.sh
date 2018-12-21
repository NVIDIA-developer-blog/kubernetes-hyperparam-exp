# create job.yaml files
for i in $(eval echo {1..$1})
do
  cat cifar10-job-template.yml | sed "s/\$ITEM/$i/" > ./hyperparam-jobs-specs/cifar10-job-$i.yml
done
