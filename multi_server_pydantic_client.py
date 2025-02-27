import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.tools import ToolDefinition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MCPServer:
    """Manages a single MCP server connection and its tools."""
    
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context = None
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = []
        self._cleanup_lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = self.config['command']
        
        server_params = StdioServerParameters(
            command=command,
            args=self.config['args'],
            env={**os.environ, **self.config['env']} if self.config.get('env') else None
        )
        
        try:
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(read, write)
            await self.session.__aenter__()
            await self.session.initialize()
            logging.info(f"Initialized MCP server: {self.name}")
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise
    
    async def load_tools(self) -> List[Dict[str, Any]]:
        """Load tools from the server."""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
            
        tools_response = await self.session.list_tools()
        self.tools = []
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    self.tools.append({
                        'name': tool.name,
                        'description': tool.description,
                        'schema': tool.inputSchema,
                        'session': self.session
                    })
                    
        logging.info(f"Loaded {len(self.tools)} tools from server: {self.name}")
        return self.tools
        
    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                if self.session:
                    try:
                        await self.session.__aexit__(None, None, None)
                    except Exception as e:
                        logging.warning(f"Warning during session cleanup for {self.name}: {e}")
                    finally:
                        self.session = None

                if self.stdio_context:
                    try:
                        await self.stdio_context.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError) as e:
                        logging.info(f"Note: Normal shutdown message for {self.name}: {e}")
                    except Exception as e:
                        logging.warning(f"Warning during stdio cleanup for {self.name}: {e}")
                    finally:
                        self.stdio_context = None
                        
                logging.info(f"Cleaned up server: {self.name}")
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


async def create_mcp_tool_executor(session: ClientSession, tool_name: str):
    """Create a function that will execute the tool on the MCP server."""
    async def mcp_tool_executor(ctx: RunContext, **kwargs):
        logging.info(f"Executing tool {tool_name} with args: {kwargs}")
        result = await session.call_tool(tool_name, kwargs)
        return result
    
    mcp_tool_executor.__name__ = tool_name
    return mcp_tool_executor


async def create_mcp_tool_preparor(tool_name: str, tool_description: str, tool_schema: Dict):
    """Create a function that will prepare MCP server tool as agent Tool for pydantic."""
    async def prepare(ctx: RunContext, tool_def: ToolDefinition):
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters_json_schema=tool_schema,
        )
        return tool_def

    return prepare

class PydanticMCPClient:
    """Client that connects to multiple MCP servers and exposes their tools via a Pydantic agent."""
    
    def __init__(self, config_path: str = 'servers_config.json') -> None:
        load_dotenv()
        self.config = self.load_config(config_path)
        self.servers: Dict[str, MCPServer] = {}
        self.agent = None
        self.tool_dict = {}
        
    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    async def initialize_servers(self) -> None:
        """Initialize all configured MCP servers."""
        for name, config in self.config['mcpServers'].items():
            server = MCPServer(name, config)
            try:
                await server.initialize()
                self.servers[name] = server
            except Exception as e:
                logging.error(f"Failed to initialize server {name}: {e}")
    
    async def load_all_tools(self) -> List[Tool]:
        """Load tools from all initialized servers and convert them to Pydantic tools."""
        pydantic_tools = []
        
        for name, server in self.servers.items():
            server_tools = await server.load_tools()
            
            for tool_info in server_tools:
                tool_name = tool_info['name']

                executor = await create_mcp_tool_executor(
                    tool_info['session'], 
                    tool_name
                )
                
                preparor = await create_mcp_tool_preparor(tool_name, tool_info['description'], tool_info['schema'])
                
                mcp_tool = Tool(executor, prepare=preparor)
                pydantic_tools.append(mcp_tool)
                
                self.tool_dict[tool_name] = {
                    'description': tool_info['description'], 
                    'schema': tool_info['schema']
                }
        
        return pydantic_tools
    
    async def create_agent(self, model_name: str = 'claude-3-5-sonnet-latest') -> Agent:
        """Create a Pydantic agent with tools from all servers."""
        if not self.servers:
            raise RuntimeError("No servers initialized. Call initialize_servers() first.")
            
        pydantic_tools = await self.load_all_tools()
        
        model = AnthropicModel(model_name)
        self.agent = Agent(model, deps_type=str, tools=pydantic_tools)
        
        # Debug: Check what tools are actually registered
        logging.info(f"Registered tool names: {list(self.agent._function_tools.keys())}")
        logging.info(f"Expected tool names: {list(self.tool_dict.keys())}")
        
        # Fix tool schemas and descriptions - preparor doesn't fix this for some reason
        for tool_name, tool_obj in self.tool_dict.items():
            if tool_name in self.agent._function_tools:
                self.agent._function_tools[tool_name]._parameters_json_schema = tool_obj['schema']
                self.agent._function_tools[tool_name].description = tool_obj['description']
            else:
                logging.warning(f"Tool '{tool_name}' not found in registered tools")
            
        # logging.info("Registered Tools")
        # for tool_name, tool_obj in self.agent._function_tools.items():
        #     logging.info(f"{tool_name} \t\t {tool_obj}")

        return self.agent
    
    async def run_query(self, query: str) -> str:
        """Run a query through the agent."""
        if not self.agent:
            raise RuntimeError("Agent not created. Call create_agent() first.")
            
        result = await self.agent.run(query)
        return result
    
    async def cleanup(self) -> None:
        """Clean up all server connections."""
        # Clean up servers one by one to avoid task issues
        for name, server in self.servers.items():
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"Error during cleanup of server {name}: {e}")
        
        logging.info("All servers cleaned up")


async def main():
    """Main function to demonstrate multi-server usage."""
    client = PydanticMCPClient()
    
    try:
        await client.initialize_servers()
        
        await client.create_agent()
        
        if client.agent:
            available_tools = sorted(list(client.agent._function_tools.keys()))
            print("\nAvailable tools:")
            for tool in available_tools:
                print(f"- {tool}")
        
        print("\nEnter queries (type 'exit' to quit):")
        while True:
            user_input = input("> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue
            
            try:
                result = await client.run_query(user_input)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
            
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
