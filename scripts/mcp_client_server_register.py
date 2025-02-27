import os
import asyncio
import logging
from contextlib import asynccontextmanager

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.tools import ToolDefinition

from typing import Dict, List, Optional, Any, Union, Callable
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def create_session(server_params):
    """Create a properly managed session using async context managers."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            yield session

async def main():
    # Set up server parameters
    # server_params = StdioServerParameters(
    #     command="uvx",
    #     args=["mcp-server-git", "--repository", "."],
    #     env=os.environ
    # )

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        env=os.environ
    )
    
    print("Starting session...")
    async with create_session(server_params) as session:
        await session.initialize()
        
        # List tools
        tools_response = await session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for tool in item[1]:
                    tools.append((tool.name, tool.description, tool.inputSchema))
                    logging.info(f"{tool.name} : {tool.inputSchema}")

        
        print(f"Found {len(tools)} tools")
        for name, desc, _ in tools:
            print(f"- {name}: {desc}")
    
    print("Session closed successfully")

if __name__ == "__main__":
    asyncio.run(main())
