import boto3
import json


primary_region ="eu-west-3" #us-east-1, eu-central-1
bedrock_runtime = boto3.client("bedrock-runtime", region_name= primary_region)
inferenceProfileId = 'eu-west-3.anthropic.claude-3-5-sonnet-20240620-v1:0' 

# Example with Converse API
system_prompt = "You are an expert on AWS AI services."
input_message = "Tell me about AI service for Foundation Models"
response = bedrock_runtime.converse(
    modelId = inferenceProfileId,
    system = [{"text": system_prompt}],
    messages=[{
        "role": "user",
        "content": [{"text": input_message}]
    }]
)

print(response['output']['message']['content'])