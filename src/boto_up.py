import boto3

boto3_connection = boto3.resource('s3')

def print_s3_contents_boto3(connection):
    for bucket in connection.buckets.all():
        for key in bucket.objects.all():
            print(key.key)

print_s3_contents_boto3(boto3_connection)
