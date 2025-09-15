
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
GATEWAY_URL='https://testgateway90a1643b-tudzt74ldc.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp'
COGNITO_URL='https://agentcore-350f8e67.auth.us-west-2.amazoncognito.com'

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personal-agent")


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
 
 
def _get_cognito_token(
    cognito_domain_url: str,
    client_id: str,
    client_secret: str,
    audience: str = "MCPGateway",
) -> Dict[str, Any]:
    """
    Get OAuth2 token from Amazon Cognito or Auth0 using client credentials grant type.

    Args:
        cognito_domain_url: The full Cognito/Auth0 domain URL
        client_id: The App Client ID
        client_secret: The App Client Secret
        audience: The audience for the token (default: MCPGateway)

    Returns:
        Token response containing access_token, expires_in, token_type
    """
    # Construct the token endpoint URL
    if "auth0.com" in cognito_domain_url:
        url = f"{cognito_domain_url.rstrip('/')}/oauth/token"
        # Use JSON format for Auth0
        headers = {"Content-Type": "application/json"}
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "audience": audience,
            "grant_type": "client_credentials",
            "scope": "invoke:gateway",
        }
        # Send as JSON for Auth0
        response_method = lambda: requests.post(url, headers=headers, json=data)
    else:
        # Cognito format
        url = f"{cognito_domain_url.rstrip('/')}/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        # Send as form data for Cognito
        response_method = lambda: requests.post(url, headers=headers, data=data)

    try:
        # Make the request
        response = response_method()
        response.raise_for_status()  # Raise exception for bad status codes

        provider_type = "Auth0" if "auth0.com" in cognito_domain_url else "Cognito"
        logging.info(f"Successfully obtained {provider_type} access token")
        return response.json()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting token: {e}")
        if hasattr(response, "text") and response.text:
            logging.error(f"Response: {response.text}")
        raise

import requests
response = _get_cognito_token("https://agentcore-efb65272.auth.us-west-2.amazoncognito.com","520h7q7t90154l65dh6ieq4nbe", "1t2qram60v0oip11i3fum0f2nfgqsv1nsk2t8iqib34jnhngfkj3")

access_token =response["access_token"]

# Initialize the Gateway client
gateway_client_toolkit = GatewayClient(region_name=os.environ['AWS_DEFAULT_REGION'])
# EZ Auth - automatically sets up Cognito OAuth
# access_token = gateway_client_toolkit.get_access_token_for_cognito(cognito_result["client_info"])


def create_streamable_http_transport():
    return streamablehttp_client(GATEWAY_URL,headers={"Authorization": f"Bearer {access_token}"})

mcp_client = MCPClient(create_streamable_http_transport)

## The IAM group/user/ configured in ~/.aws/credentials should have access to Bedrock model
yourmodel = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    temperature=0.7
)


 


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
    
     
app = BedrockAgentCoreApp()
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
