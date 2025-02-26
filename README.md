# MCP Client

A basic Model Context Protocol (MCP) client implementation with example servers - one in Node.js and one in Python.

## Overview

This repository contains a simple implementation of an MCP client that can communicate with MCP-compatible servers.

## Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure servers in `servers_config.json`
e.g.
```
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "."]
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

You can get the list of available servers [!here][https://github.com/modelcontextprotocol/servers]. 

Or write your own. For that add config something like:
```
"zotero-mcp-server": {
        "command": "bash",
        "args": [
          "-c",
          "cd /Users/swair/zotero-mcp-server && source .venv/bin/activate && python -m zotero_mcp.server"
        ]
      },
```

## Running the Client

```
python client.py
```

