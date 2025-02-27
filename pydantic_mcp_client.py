import os
import json
import shutil
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.tools import ToolDefinition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def create_mcp_tool_executor(session: ClientSession, tool_name: str):
    """return a function that will execute the tool on the MCP server."""
    async def mcp_tool_executor(ctx: RunContext, **kwargs):
        # logging.info(f"running the tool {tool_name}, {kwargs}")
        # pass kwargs as actual parameters, not as a dictionary...
        result = await session.call_tool(tool_name, kwargs)
        return result
    
    mcp_tool_executor.__name__ = tool_name
    return mcp_tool_executor

async def create_mcp_tool_preparor(tool_name: str, tool_description: str, tool_schema : Dict):
    """return a function that will prepare mcp server tool as agent Tool for pydantic"""
    async def prepare(ctx: RunContext, tool_def: ToolDefinition):
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters_json_schema=tool_schema,
        )
        # logging.info(f"{tool_def}")
        return tool_def

    return prepare


async def main():
    """Main function to set up a server and run the agent."""
    load_dotenv()

    # command = shutil.which("uvx") 
    # server_params = StdioServerParameters(
    #     command="uvx",
    #     args=["mcp-server-git", "--repository", "."],
    #     env={**os.environ, **git_config['env']} if git_config.get('env') else None
    # )

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        env=os.environ,
    )

    stdio_ctx = stdio_client(server_params)
    read, write = await stdio_ctx.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()

    tools_response = await session.list_tools()
    tool_dict = {}
    pydantic_tools = []
    for item in tools_response:
        if isinstance(item, tuple) and item[0] == 'tools':
            for tool in item[1]:
                # logging.info(f"{tool.name}")
                executor = await create_mcp_tool_executor(session, tool.name)
                preparor = await create_mcp_tool_preparor(tool.name, tool.description, tool.inputSchema)
                mcp_tool = Tool(executor, prepare=preparor)

                pydantic_tools.append(mcp_tool)
                tool_dict[tool.name]  = {'description' : tool.description, 'schema' : tool.inputSchema}

    model = AnthropicModel('claude-3-5-sonnet-latest')
    agent = Agent(model, 
                  deps_type=str,
                  tools=pydantic_tools)

    # for some reason preparor doesn't fix parameters_json_schema and description, no idea why
    # so we need to fix it

    for tool_name, tool_obj in tool_dict.items():
        agent._function_tools[tool_name]._parameters_json_schema = tool_obj['schema']
        agent._function_tools[tool_name].description = tool_obj['description']

    # logging.info("Registered Tools")
    # for tool_name, tool_obj in agent._function_tools.items():
    #     logging.info(f"{tool_name} \t\t {tool_obj}")

    try: 
        # implement agent loop here: 
        result = await agent.run("open google.com")
        print(result)
    
    finally:
        await session.__aexit__(None, None, None)
        await stdio_ctx.__aexit__(None, None, None)

if __name__ == '__main__':
    asyncio.run(main())
