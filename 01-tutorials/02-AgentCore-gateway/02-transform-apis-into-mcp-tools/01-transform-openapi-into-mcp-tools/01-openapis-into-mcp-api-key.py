#!/usr/bin/env python
# coding: utf-8

# # Transform OpenAPI APIs into MCP tools using Bedrock AgentCore Gateway
# 
# ## Overview
# Customers can bring OpenAPI spec in JSON or YAML and transform the apis into MCP tools using Bedrock AgentCore Gateway. 
# 
# The Gateway workflow involves the following steps to connect your agents to external tools:
# * **Create the tools for your Gateway** - Define your tools using schemas such as OpenAPI specifications for REST APIs. The OpenAPI specifications are then parsed by Amazon Bedrock AgentCore for creating the Gateway.
# * **Create a Gateway endpoint** - Create the gateway that will serve as the MCP entry point with inbound authentication.
# * **Add targets to your Gateway** - Configure the OpenAPI targets that define how the gateway routes requests to specific tools. All the APIs that part of OpenAPI file will become an MCP-compatible tool, and will be made available through your Gateway endpoint URL. Configure outbound authorization for each OpenAPI Gateway target. 
# * **Update your agent code** - Connect your agent to the Gateway endpoint to access all configured tools through the unified MCP interface.
# 
# ![How does it work](images/openapi-gateway-apikey.png)
# 
# ### Tutorial Details
# 
# 
# | Information          | Details                                                   |
# |:---------------------|:----------------------------------------------------------|
# | Tutorial type        | Interactive                                               |
# | AgentCore components | AgentCore Gateway, AgentCore Identity                     |
# | Agentic Framework    | Strands Agents                                            |
# | Gateway Target type  | OpenAPI                                                   |
# | Agent                | Finance Agent                                        |
# | Inbound Auth IdP     | Amazon Cognito                                            |
# | Outbound Auth        | API Key                                                   |
# | LLM model            | Anthropic Claude Sonnet 3.7 Inference profile              |
# | Tutorial components  | Creating AgentCore Gateway and Invoking AgentCore Gateway |
# | Tutorial vertical    | Cross-vertical                                            |
# | Example complexity   | Easy                                                      |
# | SDK used             | boto3 , AgentCore starter kit                             |
# 
# In the first part of the tutorial we will create some AmazonCore Gateway targets
# 
# ### Tutorial Architecture
# In this tutorial we will transform operations defined in OpenAPI yaml/json file into MCP tools and host it in Bedrock AgentCore Gateway.
# The solution uses Strands Agent using Amazon Bedrock models.
# In our example we will use a strands agent which will invoke Agentcore gateway to use the tools exposed by Intrinio API

# ## Prerequisites
# 
# To execute this tutorial you will need:
# * Jupyter notebook (Python kernel)
# * uv
# * AWS credentials
# * Amazon Cognito

get_ipython().system('pip install --force-reinstall -U -r requirements.txt --quiet')


# Set some environment variables
import os
os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')
BUCKET_NAME='agentcore-gateway-251267873559-us-west-2'
FILE_NAME='intrinio-api-schema.json'
OBJECT_KEY='openapi_3_spec.json'
API_KEY='IntrinioKeyFromProvider'
 
 


import os
import sys

# Get the directory of the current script
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()  # Fallback if __file__ is not defined (e.g., Jupyter)

# Navigate to the directory containing utils.py (one level up)
utils_dir = os.path.abspath(os.path.join(current_dir, '../..'))

# Add to sys.path
sys.path.insert(0, utils_dir)

# Now you can import utils
import utils


#### Create an IAM role for the Gateway to assume
import utils

agentcore_gateway_iam_role = utils.create_agentcore_gateway_role("sample-APIgateway")
print("Agentcore gateway role ARN: ", agentcore_gateway_iam_role['Role']['Arn'])


# # Create the Cognito Authorizer & Agentcore Gateway 

from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

# Initialize the Gateway client
client = GatewayClient(region_name=os.environ['AWS_DEFAULT_REGION'])

# EZ Auth - automatically sets up Cognito OAuth
cognito_result = client.create_oauth_authorizer_with_cognito("my-gateway")


authorizer_configuration = cognito_result["authorizer_config"]


gateway = client.create_mcp_gateway(
    # name=none, # the name of the Gateway - if you don't set one, one will be generated.
    role_arn=agentcore_gateway_iam_role['Role']['Arn'], # the role arn that the Gateway will use - if you don't set one, one will be created.
    authorizer_config=authorizer_configuration, # Variable from inbound authorization setup steps. Contains the OAuth authorizer details for authorizing callers to your Gateway (MCP only supports OAuth).
    enable_semantic_search=True # enable semantic search.

)
print(f"OAuth Credentials:")
print(f"  Client ID: {cognito_result['client_info']['client_id']}")
print(f"  Scope: {cognito_result['client_info']['scope']}")
gateway_id=gateway['gatewayId']
gateway_url=gateway['gatewayUrl']


# # Transforming Intrinio Open APIs into MCP tools using Bedrock AgentCore Gateway

# We will use Intrinio APIs to expose as MCP tools. We will use Intrinio API key to configure the credentials provider for creating the OpenAPI target.

import boto3
import json
from pprint import pprint
from botocore.config import Config
import boto3
from botocore.exceptions import ClientError

client = boto3.client('secretsmanager', region_name=os.environ['AWS_DEFAULT_REGION'])
response = client.get_secret_value(SecretId=API_KEY)
secret_dict = json.loads(response['SecretString'])
secret_value = list(secret_dict.values())[0]
acps = boto3.client(service_name="bedrock-agentcore-control")

try:
    response= acps.create_api_key_credential_provider(
        name="IntrinioAPIKey",
        apiKey=secret_value,  
    )
except Exception as e:

    print(e)
    

    response = acps.get_api_key_credential_provider(
        name="IntrinioAPIKey"
    )
credentialProviderARN = response['credentialProviderArn']
pprint(f"Egress Credentials provider ARN, {credentialProviderARN}")


# #### If you see an error as below
# #### "An error occurred (ValidationException) when calling the CreateApiKeyCredentialProvider operation: Credential provider with #### name: IntrinioAPIKey already exists
# #### ('Egress Credentials provider ARN, '
# #### 'arn:aws:bedrock-agentcore:xxxxxxxxxx:token-vault/default/apikeycredentialprovider/IntrinioAPIKey')"
# ####  ignore the error. This means the credential provider is created by other users

# # Create an OpenAPI target 

# #### We will use a S3 bucket to store the OpenAPI spec from Intrinio

openapi_s3_uri = f's3://{BUCKET_NAME}/{OBJECT_KEY}'
print(f'Uploaded object S3 URI: {openapi_s3_uri}')


# #### Configure outbound auth and Create the gateway target

gateway_client = boto3.client('bedrock-agentcore-control', region_name = os.environ['AWS_DEFAULT_REGION'])

# S3 Uri for OpenAPI spec file
Intrinio_openapi_s3_target_config = {
    "mcp": {
          "openApiSchema": {
              "s3": {
                  "uri": openapi_s3_uri
              }
          }
      }
}

# API Key credentials provider configuration
api_key_credential_config = [
    {
        "credentialProviderType" : "API_KEY", 
        "credentialProvider": {
            "apiKeyCredentialProvider": {
                    "credentialParameterName": "api_key", # Replace this with the name of the api key name expected by the respective API provider. For passing token in the header, use "Authorization"
                    "providerArn": credentialProviderARN,
                    "credentialLocation":"QUERY_PARAMETER", # Location of api key. Possible values are "HEADER" and "QUERY_PARAMETER".
                    #"credentialPrefix": " " # Prefix for the token. Valid values are "Basic". Applies only for tokens.
            }
        }
    }
  ]

targetname='DemoOpenAPITargetS3Intrinio'
response = gateway_client.create_gateway_target(
    gatewayIdentifier=gateway_id,
    name=targetname,
    description='OpenAPI Target with S3Uri using SDK',
    targetConfiguration=Intrinio_openapi_s3_target_config,
    credentialProviderConfigurations=api_key_credential_config)


# # Calling Bedrock AgentCore Gateway from a Strands Agent
# 
# The Strands agent seamlessly integrates with AWS tools through the Bedrock AgentCore Gateway, which implements the Model Context Protocol (MCP) specification. This integration enables secure, standardized communication between AI agents and AWS services.
# 
# At its core, the Bedrock AgentCore Gateway serves as a protocol-compliant Gateway that exposes fundamental MCP APIs: ListTools and InvokeTools. These APIs allow any MCP-compliant client or SDK to discover and interact with available tools in a secure, standardized way. When the Strands agent needs to access AWS services, it communicates with the Gateway using these MCP-standardized endpoints.
# 
# The Gateway's implementation adheres strictly to the (MCP Authorization specification)[https://modelcontextprotocol.org/specification/draft/basic/authorization], ensuring robust security and access control. This means that every tool invocation by the Strands agent goes through authorization step, maintaining security while enabling powerful functionality.
# 
# For example, when the Strands agent needs to access MCP tools, it first calls ListTools to discover available tools, then uses InvokeTools to execute specific actions. The Gateway handles all the necessary security validations, protocol translations, and service interactions, making the entire process seamless and secure.
# 
# This architectural approach means that any client or SDK that implements the MCP specification can interact with AWS services through the Gateway, making it a versatile and future-proof solution for AI agent integrations.

# # Request the access token from Amazon Cognito for inbound authorization

from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

# Initialize the Gateway client
gateway_client_toolkit = GatewayClient(region_name=os.environ['AWS_DEFAULT_REGION'])
# EZ Auth - automatically sets up Cognito OAuth
access_token = gateway_client_toolkit.get_access_token_for_cognito(cognito_result["client_info"])


# # Finance agent will use Bedrock AgentCore Gateway to retrive information from MCP tools

from strands.models import BedrockModel
from mcp.client.streamable_http import streamablehttp_client 
from strands.tools.mcp.mcp_client import MCPClient
from strands import Agent

def create_streamable_http_transport():
    return streamablehttp_client(gateway_url,headers={"Authorization": f"Bearer {access_token}"})

mcp_client = MCPClient(create_streamable_http_transport)

## The IAM group/user/ configured in ~/.aws/credentials should have access to Bedrock model
yourmodel = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    temperature=0.7
)


from strands import Agent
import logging


SYSTEM_PROMPT="You are a Financial Agent. You can use various tools available to you to get the financial and company information for a company" \
"Use the company name or ticker within the prompt and pass it as a required parametr or identifier to the tools. Identify the required parameters or identifiers" \
"Sometimes tag is a required parameter to the tool . use your judgement to derive a possible value for the tag from the prompt" 
# Configure the root strands logger. Change it to DEBUG if you are debugging the issue
logging.getLogger("strands").setLevel(logging.INFO)

# Add a handler to see the logs
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)

with mcp_client:
    # Call the listTools 
    tools = mcp_client.list_tools_sync()
    # Create an Agent with the model and tools
    agent = Agent(model=yourmodel,tools=tools, system_prompt=SYSTEM_PROMPT) ## you can replace with any model you like
    print(f"Tools loaded in the agent are {agent.tool_names}")
    # print(f"Tools configuration in the agent are {agent.tool_config}")
    # Invoke the agent with the sample prompt. This will only invoke  MCP listTools and retrieve the list of tools the LLM has access to. The below does not actually call any tool.
    # agent("Hi , can you list all tools available to you")
    agent("get company information for Nvidia")
    agent("get company financial information for Apple")
    # Invoke the agent with sample prompt, invoke the tool and display the response
    #Call the MCP tool explicitly. The MCP Tool name and arguments must match with your AWS Lambda function or the OpenAPI/Smithy API
    # result = client.call_tool_sync(
    # tool_use_id="get-intrinio_tools_1", # You can replace this with unique identifier. 
    # name=targetname+"___getCompanyFundamentals", # This is the tool name based on AWS Lambda target types. This will change based on the target name
    # arguments={"ver": "1.0","feedtype": "json"}
    #)
    #Print the MCP Tool response
    #print(f"Tool Call result: {result['content'][0]['text']}")


# # Strands Agents with AgentCore Memory (Short-Term Memory)
# 
# 
# ## Introduction
# 
# This tutorial demonstrates how to build a **personal agent** using Strands agents with AgentCore **short-term memory** (Raw events). The agent remembers recent conversations in the session using `get_last_k_turns` and can continue conversations seamlessly when user returns.
# 
# 
# ### Tutorial Details
# 
# | Information         | Details                                                                          |
# |:--------------------|:---------------------------------------------------------------------------------|
# | Tutorial type       | Short Term Conversational                                                        |
# | Agent type          | Personal Agent                                                                   |
# | Agentic Framework   | Strands Agents                                                                   |
# | LLM model           | Anthropic Claude Sonnet 3.7                                                      |
# | Tutorial components | AgentCore Short-term Memory, AgentInitializedEvent and MessageAddedEvent hooks   |
# | Example complexity  | Beginner                                                                         |
# 
# You'll learn to:
# - Use short-term memory for conversation continuity
# - Retrieve last K conversation turns
# - Web search tool for real-time information
# - Initialize agents with conversation history
# 
# ## Architecture
# <div style="text-align:left">
#     <img src="architecture.png" width="65%" />
# </div>
# 
# ## Prerequisites
# 
# - Python 3.10+
# - AWS credentials with AgentCore Memory permissions
# - AgentCore Memory role ARN
# - Access to Amazon Bedrock models
# 
# Let's get started by setting up our environment!

import logging
from datetime import datetime

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personal-agent")


# Imports
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from bedrock_agentcore.memory import MemoryClient

# Configuration
REGION = os.getenv('AWS_REGION', 'us-east-1') # AWS region for the agent
ACTOR_ID = "user_123" # It can be any unique identifier (AgentID, User ID, etc.)
SESSION_ID = "personal_session_001" # Unique session identifier


from botocore.exceptions import ClientError
import uuid

# Initialize Memory Client
client = MemoryClient(region_name=REGION)
memory_name = f"PersonalAgentMemory_{uuid.uuid4().hex[:8]}"

try:
    # Create memory resource without strategies (thus only access to short-term memory)
    memory = client.create_memory_and_wait(
        name=memory_name,
        strategies=[],  # No strategies for short-term memory
        description="Short-term memory for personal agent",
        event_expiry_days=7, # Retention period for short-term memory. This can be upto 365 days.
    )
    memory_id = memory['id']
    logger.info(f"✅ Created memory: {memory_id}")
except ClientError as e:
    logger.info(f"❌ ERROR: {e}")
    if e.response['Error']['Code'] == 'ValidationException' and "already exists" in str(e):
        # If memory already exists, retrieve its ID
        memories = client.list_memories()
        memory_id = next((m['id'] for m in memories if m['id'].startswith(memory_name)), None)
        logger.info(f"Memory already exists. Using existing memory ID: {memory_id}")
except Exception as e:
    # Show any errors during memory creation
    logger.error(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    # Cleanup on error - delete the memory if it was partially created
    if memory_id:
        try:
            client.delete_memory_and_wait(memory_id=memory_id)
            logger.info(f"Cleaned up memory: {memory_id}")
        except Exception as cleanup_error:
            logger.error(f"Failed to clean up memory: {cleanup_error}")


class MemoryHookProvider(HookProvider):
    def __init__(self, memory_client: MemoryClient, memory_id: str, actor_id: str, session_id: str):
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
    
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load recent conversation history when agent starts"""
        try:
            # Load the last 5 conversation turns from memory
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=5
            )
            
            if recent_turns:
                # Format conversation history for context
                context_messages = []
                for turn in recent_turns:
                    for message in turn:
                        role = message['role']
                        content = message['content']['text']
                        context_messages.append(f"{role}: {content}")
                
                context = "\n".join(context_messages)
                # Add context to agent's system prompt.
                event.agent.system_prompt += f"\n\nRecent conversation:\n{context}"
                logger.info(f"✅ Loaded {len(recent_turns)} conversation turns")
                
        except Exception as e:
            logger.error(f"Memory load error: {e}")
    
    def on_message_added(self, event: MessageAddedEvent):
        """Store messages in memory"""
        messages = event.agent.messages
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(str(messages[-1].get("content", "")), messages[-1]["role"])]
            )
        except Exception as e:
            logger.error(f"Memory save error: {e}")
    
    def register_hooks(self, registry: HookRegistry):
        # Register memory hooks
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)


with mcp_client:
    tools = mcp_client.list_tools_sync()


def create_personal_agent():
    """Create personal agent with memory and web search"""
    agent = Agent(
        name="PersonalAssistant",
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # or your preferred model
        system_prompt=f""" You are a Financial Agent. You can use various tools available to you to get the financial and company information for a company
Use the company name or ticker within the prompt and pass it as a required parametr or identifier to the tools. Identify the required parameters or identifiers
Sometimes tag is a required parameter to the tool . use your judgement to derive a possible value for the tag from the prompt
        
       
        
       
        Today's date: {datetime.today().strftime('%Y-%m-%d')}
        Be friendly and professional.""",
        hooks=[MemoryHookProvider(client, memory_id, ACTOR_ID, SESSION_ID)],
        tools=tools,
    )
    return agent

# Create agent
agent = create_personal_agent()
logger.info("✅ Personal agent created with memory and web search")


with mcp_client:
#    agent("get company information for Nvidia")
#   agent("My name is Alex and I'm interested in learning about company IBM.")
    agent("I'm particularly interested in machine learning applications.")


# Create new agent instance (simulates user returning)
print("=== User Returns - New Session ===")
new_agent = create_personal_agent()

# Test memory continuity
print(f"User: What was my name again?")
print(f"Agent: ", end="")
with mcp_client:
   new_agent("What was my name again?")

   print(f"User: What was my last question?")
   print(f"Agent: ", end="")
   new_agent("what was my last question")


memory_id


# ## View Stored Memory

# Check what's stored in memory
print("=== Memory Contents ===")
recent_turns = client.get_last_k_turns(
    memory_id=memory_id,
    actor_id=ACTOR_ID,
    session_id=SESSION_ID,
    k=3 # Adjust k to see more or fewer turns
)

for i, turn in enumerate(recent_turns, 1):
    print(f"Turn {i}:")
    for message in turn:
        role = message['role']
        content = message['content']['text'][:100] + "..." if len(message['content']['text']) > 100 else message['content']['text']
        print(f"  {role}: {content}")
    print()





get_ipython().run_cell_magic('writefile', 'finance_agent_claude.py', 'from bedrock_agentcore.runtime import BedrockAgentCoreApp\n\napp = BedrockAgentCoreApp()\ndef create_personal_agent():\n    """Create personal agent with memory and web search"""\n    agent = Agent(\n        name="PersonalAssistant",\n        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # or your preferred model\n        system_prompt=f""" You are a Financial Agent. You can use various tools available to you to get the financial and company information for a company\nUse the company name or ticker within the prompt and pass it as a required parametr or identifier to the tools. Identify the required parameters or identifiers\nSometimes tag is a required parameter to the tool . use your judgement to derive a possible value for the tag from the prompt\n        \n       \n        \n       \n        Today\'s date: {datetime.today().strftime(\'%Y-%m-%d\')}\n        Be friendly and professional.""",\n        hooks=[MemoryHookProvider(client, memory_id, ACTOR_ID, SESSION_ID)],\n        tools=tools,\n    )\n    return agent\nagent = create_personal_agent()\nlogger.info("✅ Personal agent created with memory and web search")\n\n@app.entrypoint\ndef strands_agent_bedrock(payload):\n    """\n    Invoke the agent with a payload\n    """\n    user_input = payload.get("prompt")\n    print("User input:", user_input)\n    response = agent(user_input)\n    return response.message[\'content\'][0][\'text\']\n\nif __name__ == "__main__":\n    app.run()\n')


# # Clean up
# Additional resources are also created like IAM role, IAM Policies, Credentials provider, AWS Lambda functions, Cognito user pools, s3 buckets that you might need to manually delete as part of the clean up. This depends on the example you run.

# ## Delete the gateway (Optional)

import utils
utils.delete_gateway(gateway_client,gateway_id)
client.delete_memory_and_wait(memory_id)
logger.info(f"✅ Deleted memory: {memory_id}")




