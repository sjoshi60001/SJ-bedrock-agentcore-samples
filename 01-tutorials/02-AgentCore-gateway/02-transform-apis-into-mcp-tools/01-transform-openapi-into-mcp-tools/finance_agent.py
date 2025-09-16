
import boto3
from strands.models import BedrockModel
from mcp.client.streamable_http import streamablehttp_client 
from strands.tools.mcp.mcp_client import MCPClient
from strands import Agent
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
import logging
from datetime import datetime
import os

# Imports
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from bedrock_agentcore.memory import MemoryClient

# Configuration
REGION = os.getenv('AWS_REGION', 'us-east-1') # AWS region for the agent
ACTOR_ID = "user_123" # It can be any unique identifier (AgentID, User ID, etc.)
SESSION_ID = "personal_session_001" # Unique session identifier
GATEWAY_URL='https://testgateway810d5a68-sbcqd21fmi.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp'

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance-agent")


from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient
from typing import Dict, Any
from botocore.exceptions import ClientError
import uuid

# Configuration
os.environ['AWS_DEFAULT_REGION'] = os.environ.get('AWS_REGION', 'us-east-1')
ACTOR_ID = "user_123" # It can be any unique identifier (AgentID, User ID, etc.)
SESSION_ID = "personal_session_001" # Unique session identifier
# Initialize Memory Client
from lab_helpers.utils import get_ssm_parameter, put_ssm_parameter  
memory_client = MemoryClient(region_name=os.environ['AWS_DEFAULT_REGION'])
memory_name = "FinanceAgentMemory"

def create_or_get_memory_resource():
    try:
        memory_id = get_ssm_parameter("/app/financeagent/agentcore/memory_id")
        memory_client.gmcp_client.get_memory(memoryId=memory_id)
        return memory_id
    except:
        try:
            print("Creating AgentCore Memory resources. This will take 2-3 minutes...")
            print("While we wait, let's understand what's happening behind the scenes:")
            print("• Setting up managed vector databases for semantic search")
            print("• Configuring memory extraction pipelines")
            print("• Provisioning secure, multi-tenant storage")
            print("• Establishing namespace isolation for customer data")
            # *** AGENTCORE MEMORY USAGE *** - Create memory resource with semantic strategy
            response = memory_client.create_memory_and_wait(
                name=memory_name,
                description="short term  memory for finance agent",
                strategies=[],
                event_expiry_days=90,          # Memories expire after 90 days
            )
            memory_id = response["id"]
            try:
                put_ssm_parameter("/app/financeagent/agentcore/memory_id", memory_id)
            except:
                raise
            return memory_id
        except Exception as e:
            print(f"Failed to create memory resource: {e}")
            return None
            
memory_id = create_or_get_memory_resource()
if memory_id:
    print("✅ AgentCore Memory created successfully!")
    print(f"Memory ID: {memory_id}")
else:
    print("Memory resource not created. Try Again !")
    
class MemoryHookProvider(HookProvider):
    def __init__(self, memory_client: memory_client, memory_id: str, actor_id: str, session_id: str):
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
        MAX_LENGTH = 9000
        """Store messages in memory"""
        messages = event.agent.messages     
        if len(messages) > MAX_LENGTH:
            truncated_text = messages[:MAX_LENGTH]
        else:
            truncated_text = messages
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(str(truncated_text[-1].get("content", "")), truncated_text[-1]["role"])]
            )
        except Exception as e:
            logger.error(f"Memory save error: {e}")
    
    def register_hooks(self, registry: HookRegistry):
        # Register memory hooks
        registry.add_callback(MessageAddedEvent, self.on_message_added)

import base64, json, time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def get_secret():

    secret_name = "AthenzPrivateKey"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']
    return secret

def generate_jwt_token():

    # JWT header and payload
    header = {
        "alg": "ES256",
        "typ": "JWT",
        "kid": "v0"
    }
    payload = {
        "iss": "idb2b.finance.agentic-test.217f928e-f18b-4123-9682-dd320fc1fcb4",
        "sub": "idb2b.finance.agentic-test.217f928e-f18b-4123-9682-dd320fc1fcb4",
        "aud": "https://id-uat.b2b.yahooincapis.com/zts/v1", 
        "exp": int(time.time()) + 10 * 60 * 60  # 10 minutes
    }
    
    # Encode header and payload
    encoded_header = b64url(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}".encode()
    private_key_str = get_secret()
    private_key = serialization.load_pem_private_key(bytearray(private_key_str, "UTF-8"), password=None)
    # Load EC private key
  #  with open("./client_private_key.pem", "rb") as key_file:
#     private_key = serialization.load_pem_private_key(key_file.read(), password=None)
    print ("private key successful")
    # Sign and convert DER → raw (r||s)
    der_signature = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_signature)
    r_bytes = r.to_bytes(32, byteorder="big")
    s_bytes = s.to_bytes(32, byteorder="big")
    raw_signature = r_bytes + s_bytes
    
    # Encode raw signature to base64url
    encoded_signature = b64url(raw_signature)
    
    # Final JWT
    jwt_token = f"{encoded_header}.{encoded_payload}.{encoded_signature}"
    print("JWT Client Assertion:\n", jwt_token)
    return jwt_token

def get_access_token():
    url = "https://id-uat.b2b.yahooincapis.com/zts/v1/oauth2/token"
    payload = {
        "grant_type": "client_credentials",
        "scope": "agent",
        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "client_assertion": generate_jwt_token()
    }
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    response = requests.post(url, data=payload, headers=headers)
    
   
    response_json=response.json()
    access_token = response_json.get("access_token")
    print("access token: ", access_token)
    return access_token


# Initialize the Gateway client
gateway_client_toolkit = GatewayClient(region_name=os.environ['AWS_DEFAULT_REGION'])
 
import requests
from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

# Initialize the Gateway client
gateway_client_toolkit = GatewayClient(region_name=os.environ['AWS_DEFAULT_REGION'])

access_token =  get_access_token()

def create_streamable_http_transport():
    return streamablehttp_client(GATEWAY_URL,headers={"Authorization": f"Bearer {access_token}"})

mcp_client = MCPClient(create_streamable_http_transport)

## The IAM group/user/ configured in ~/.aws/credentials should have access to Bedrock model
 
with mcp_client:
    # Call the listTools 
    tools = mcp_client.list_tools_sync()
    
     
app = BedrockAgentCoreApp()

def create_personal_agent():
    """Create personal agent with memory """
    agent = Agent(
        name="PersonalAssistant",
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        system_prompt=f"""You are a Financial Agent. You can use various tools available to you to get the financial and company information for a company
                        Use the company name or ticker within the prompt and pass it as a required parametr or identifier to the tools. Identify the required parameters or identifiers
                        Sometimes tag is a required parameter to the tool . use your judgement to derive a possible value for the tag from the prompt          
                        Today's date: {datetime.today().strftime('%Y-%m-%d')}
                        Be friendly and professional.""",
        hooks=[MemoryHookProvider(memory_client, memory_id, ACTOR_ID, SESSION_ID)],
        tools=tools,
    )
    return agent

agent = create_personal_agent()
logger.info("✅ Personal agent created with memory and web search")

@app.entrypoint
def strands_agent_bedrock(payload):
    """
    Invoke the agent with a payload
    """
    user_input = payload.get("prompt")
    print("User input:", user_input)
    with mcp_client:
        response = agent(user_input)
        return response.message['content'][0]['text']

if __name__ == "__main__":
    app.run()
