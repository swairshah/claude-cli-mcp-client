import asyncio
import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.
        
        Args:
            file_path: Path to the JSON configuration file.
            
        Returns:
            Dict containing server configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.
        
        Returns:
            The API key as a string.
            
        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.stdio_context: Optional[Any] = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.capabilities: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = shutil.which("npx") if self.config['command'] == "npx" else self.config['command']
        if command is None:
            raise ValueError(f"Command not found: {self.config['command']}")
            
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
            initialize_result = await self.session.initialize()
            self.capabilities = {}
            if hasattr(initialize_result, 'capabilities'):
                self.capabilities = initialize_result.capabilities
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """List available tools from the server.
        
        Returns:
            A list of available tools.
            
        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        
        tools_response = await self.session.list_tools()
        tools = []
        
        supports_progress = (
            self.capabilities 
            and 'progress' in self.capabilities
        )
        
        if supports_progress:
            logging.info(f"Server {self.name} supports progress tracking")
        
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
                    if supports_progress:
                        logging.info(f"Tool '{tool.name}' will support progress tracking")
        
        return tools

    async def execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any], 
        retries: int = 2, 
        delay: float = 1.0
    ) -> Any:
        """Execute a tool with retry mechanism.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.
            
        Returns:
            Tool execution result.
            
        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                supports_progress = (
                    self.capabilities 
                    and 'progress' in self.capabilities
                )

                if supports_progress:
                    logging.info(f"Executing {tool_name} with progress tracking...")
                    # Check if call_tool supports progress_token parameter
                    try:
                        result = await self.session.call_tool(
                            tool_name, 
                            arguments,
                            progress_token=f"{tool_name}_execution"
                        )
                    except TypeError:
                        # Fall back to standard call if progress_token is not supported
                        logging.info("Progress tracking not supported by call_tool, using standard call")
                        result = await self.session.call_tool(tool_name, arguments)
                else:
                    logging.info(f"Executing {tool_name}...")
                    result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

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
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if 'properties' in self.input_schema:
            for param_name, param_info in self.input_schema['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider (Anthropic)."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.client = AsyncAnthropic(api_key=api_key)

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            Exception: If the request to the LLM fails.
        """
        try:
            anthropic_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    system_message = content
                    continue
                elif role == "user":
                    anthropic_role = "user"
                elif role == "assistant":
                    anthropic_role = "assistant"
                else:
                    # Skip unknown roles
                    continue
                
                anthropic_messages.append({"role": anthropic_role, "content": content})
            
            # Create a non-streaming message
            response = await self.client.messages.create(
                max_tokens=4096,
                messages=anthropic_messages,
                model="claude-3-5-sonnet-latest",
                system=system_message if 'system_message' in locals() else None,
            )
            
            return response.content[0].text
            
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
    
    async def get_streaming_response(self, messages: List[Dict[str, str]], print_stream: bool = True) -> str:
        """Get a streaming response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            print_stream: Whether to print the response as it streams in.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            Exception: If the request to the LLM fails.
        """
        try:
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "system":
                    system_message = content
                    continue
                elif role == "user":
                    anthropic_role = "user"
                elif role == "assistant":
                    anthropic_role = "assistant"
                else:
                    continue
                
                anthropic_messages.append({"role": anthropic_role, "content": content})
            
            stream = await self.client.messages.create(
                max_tokens=4096,
                messages=anthropic_messages,
                model="claude-3-5-sonnet-latest",
                system=system_message,
                stream=True,
            )
            
            if print_stream:
                print("\nAssistant: ", end="", flush=True)
                
            # Collect the response
            full_response = ""
            async for event in stream:
                if event.type == "content_block_delta":
                    full_response += event.delta.text
                    # Print progress if requested
                    if print_stream:
                        print(event.delta.text, end="", flush=True)
            
            if print_stream:
                print()
                
            return full_response
            
        except Exception as e:
            error_message = f"Error getting streaming LLM response: {str(e)}"
            logging.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[Server], llm_client: LLMClient, use_streaming: bool) -> None:
        self.servers: List[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.use_streaming = use_streaming

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))
        
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.
        
        Args:
            llm_response: The response from the LLM.
            
        Returns:
            The result of tool execution or the original response.
        """
        import json
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")
                
                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                            
                            if isinstance(result, dict) and 'progress' in result:
                                progress = result['progress']
                                total = result['total']
                                logging.info(f"Progress: {progress}/{total} ({(progress/total)*100:.1f}%)")
                                
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg
                
                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return
            
            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)
            
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
            
            system_message = f"""You are a helpful assistant with access to these tools: 

{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""

            messages = [
                {
                    "role": "system",
                    "content": system_message
                }
            ]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ['quit', 'exit']:
                        logging.info("\nExiting...")
                        break

                    messages.append({"role": "user", "content": user_input})
                    
                    if self.use_streaming:
                        llm_response = await self.llm_client.get_streaming_response(messages)
                    else:
                        llm_response = await self.llm_client.get_response(messages)
                        logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)
                    
                    if result != llm_response:
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})
                        
                        if self.use_streaming:
                            print("\nFinal response: ", end="", flush=True)
                            final_response = await self.llm_client.get_streaming_response(messages, print_stream=True)
                        else:
                            final_response = await self.llm_client.get_response(messages)
                            logging.info("\nFinal response: %s", final_response)
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break
        
        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config('servers_config.json')
    servers = [Server(name, srv_config) for name, srv_config in server_config['mcpServers'].items()]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client, use_streaming=False)
    await chat_session.start()

if __name__ == "__main__":
    asyncio.run(main())
