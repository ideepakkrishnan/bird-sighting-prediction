# Makefile for MapReduce Climate Analysis program.

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/lib/spark-2.1.0-bin-hadoop2.7
jar.name=predictor-1.0-SNAPSHOT-jar-with-dependencies.jar
jar.path=target/${jar.name}
job.name=com.neu.pdp.App
# Local execution options
local.gbtModelInput=/home/ideepakkrishnan/Documents/project/aws-pipeline-10m-12d/GradientBoostModel
local.lrModelInput=/home/ideepakkrishnan/Documents/project/aws-pipeline-10m-12d/LogisticRegressionModel
local.input=/home/ideepakkrishnan/Documents/project/unlabeledInput
local.output=/home/ideepakkrishnan/Documents/project/predictions
# AWS EMR Execution
aws.emr.release=emr-5.2.1
aws.region=us-east-1
aws.bucket.name=pdpbucket
aws.subnet.id=subnet-b6f4c5ff
aws.gbtModelInput=output/GradientBoostModel
aws.lrModelInput=output/LogisticRegressionModel
aws.input=unlabeledInput
aws.output=predictions
aws.log.dir=logs
aws.num.nodes=3
aws.instance.type=m4.large
# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
#	mvn clean package

# Removes local output directory.
clean-local-output:
	rm -rf ${local.output}*

# Runs standalone
alone: jar clean-local-output
	${spark.root}/bin/spark-submit --class ${job.name} --master local[*] ${jar.path} ${local.lrModelInput} ${local.gbtModelInput} ${local.input} ${local.output}

# Upload data to S3 input dir.
upload-input-aws: 
#	aws s3 sync ${local.input} s3://${aws.bucket.name}/${aws.input}
	
# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.path} s3://${aws.bucket.name}

# Main EMR launch.
cloud: jar upload-input-aws upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "Predictor Cluster" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
	    --applications Name=Spark \
	    --steps '[{"Name":"Sighting Prediction Training", "Args":["--class", "${job.name}", "--master", "yarn", "--deploy-mode", "cluster", "s3://${aws.bucket.name}/${jar.name}", "s3://${aws.bucket.name}/${aws.lrModelInput}", "s3://${aws.bucket.name}/${aws.gbtModelInput}", "s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}"],"Type":"Spark","Jar":"s3://${aws.bucket.name}/${jar.name}","ActionOnFailure":"TERMINATE_CLUSTER"}]' \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--configurations '[{"Classification":"spark", "Properties":{"maximizeResourceAllocation": "true"}}]' \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate
