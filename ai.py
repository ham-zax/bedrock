import boto3
import json

primary_region = "eu-west-3"
bedrock_runtime = boto3.client("bedrock-runtime", region_name=primary_region)
inferenceProfileId = 'eu.anthropic.claude-3-5-sonnet-20240620-v1:0'

system_prompt = "You are an expert on AWS AI services."
input_message = "Tell me about AI service for Foundation Models"

try:
    response = bedrock_runtime.converse(
        modelId=inferenceProfileId,
        system=[{"text": system_prompt}],
        messages=[{
            "role": "user",
            "content": [{"text": input_message}]
        }]
    )
    print(response['output']['message']['content'])
except Exception as e:
    print(f"An error occurred: {str(e)}")