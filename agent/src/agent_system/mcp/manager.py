"""
MCP Tools集成模块
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from agent_system.core.config import MCPConfig
from agent_system.core.exceptions import MCPError, MCPToolNotFoundError
from agent_system.core.logging import get_logger

logger = get_logger(__name__)


try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPToolWrapper(BaseTool):
    """MCP工具包装器"""
    
    server_name: str
    tool_name: str
    tool_description: str
    input_schema: Dict[str, Any]
    
    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_description: str,
        input_schema: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        args_schema = self._create_args_schema(input_schema)
        
        super().__init__(
            name=f"{server_name}_{tool_name}",
            description=tool_description,
            args_schema=args_schema,
            server_name=server_name,
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            **kwargs,
        )
    
    def _create_args_schema(self, input_schema: Dict[str, Any]) -> Type[BaseModel]:
        """从JSON Schema创建Pydantic模型"""
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        fields = {}
        for prop_name, prop_schema in properties.items():
            prop_type = self._get_python_type(prop_schema.get("type", "string"))
            prop_description = prop_schema.get("description", "")
            is_required = prop_name in required
            
            if is_required:
                fields[prop_name] = (prop_type, Field(..., description=prop_description))
            else:
                default = prop_schema.get("default", None)
                fields[prop_name] = (prop_type, Field(default, description=prop_description))
        
        return create_model(f"{self.name}Args", **fields)
    
    def _get_python_type(self, json_type: str) -> Type:
        """将JSON类型转换为Python类型"""
        type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)
    
    def _run(self, **kwargs: Any) -> str:
        """同步执行工具"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs: Any) -> str:
        """异步执行工具"""
        raise NotImplementedError(
            "MCPToolWrapper must be used with MCPManager to execute tools"
        )


class MCPServerConnection:
    """MCP服务器连接"""
    
    def __init__(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._session: Optional[Any] = None
        self._client: Optional[Any] = None
        self._tools: List[Dict[str, Any]] = []
    
    async def connect(self) -> None:
        """连接到MCP服务器"""
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP is not installed. Install with: pip install mcp mcp-client"
            )
        
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        
        self._client = stdio_client(server_params)
        read, write = await self._client.__aenter__()
        
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        
        await self._session.initialize()
        
        tools_response = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else dict(tool.inputSchema),
            }
            for tool in tools_response.tools
        ]
        
        logger.info(
            "Connected to MCP server",
            server=self.server_name,
            tools=[t["name"] for t in self._tools],
        )
    
    async def disconnect(self) -> None:
        """断开MCP服务器连接"""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if self._client:
            await self._client.__aexit__(None, None, None)
        
        self._session = None
        self._client = None
        
        logger.info("Disconnected from MCP server", server=self.server_name)
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """执行MCP工具"""
        if not self._session:
            raise MCPError(
                f"Not connected to MCP server: {self.server_name}",
                self.server_name,
            )
        
        result = await self._session.call_tool(
            tool_name=tool_name,
            arguments=arguments,
        )
        
        content_parts = []
        for content in result.content:
            if hasattr(content, 'text'):
                content_parts.append(content.text)
            elif hasattr(content, 'model_dump'):
                content_parts.append(json.dumps(content.model_dump()))
            else:
                content_parts.append(str(content))
        
        return "\n".join(content_parts)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具"""
        return self._tools
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._session is not None


class MCPManager:
    """
    MCP管理器
    
    管理多个MCP服务器连接，提供统一的工具调用接口
    """
    
    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self._connections: Dict[str, MCPServerConnection] = {}
        self._tool_wrappers: Dict[str, MCPToolWrapper] = {}
        self._tool_to_server: Dict[str, str] = {}
    
    async def connect_all(self) -> None:
        """连接所有配置的MCP服务器"""
        for server_config in self.config.servers:
            server_name = server_config.get("name")
            if not server_name:
                logger.warning("MCP server config missing 'name', skipping")
                continue
            
            command = server_config.get("command")
            if not command:
                logger.warning(
                    "MCP server config missing 'command', skipping",
                    server=server_name,
                )
                continue
            
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            
            connection = MCPServerConnection(
                server_name=server_name,
                command=command,
                args=args,
                env=env,
            )
            
            try:
                await connection.connect()
                self._connections[server_name] = connection
                
                for tool in connection.get_tools():
                    wrapper = MCPToolWrapper(
                        server_name=server_name,
                        tool_name=tool["name"],
                        tool_description=tool["description"],
                        input_schema=tool["input_schema"],
                    )
                    full_tool_name = f"{server_name}_{tool['name']}"
                    self._tool_wrappers[full_tool_name] = wrapper
                    self._tool_to_server[tool["name"]] = server_name
                
                logger.info(
                    "MCP server connected",
                    server=server_name,
                    tools=[t["name"] for t in connection.get_tools()],
                )
            except Exception as e:
                logger.error(
                    "Failed to connect to MCP server",
                    server=server_name,
                    error=str(e),
                )
    
    async def disconnect_all(self) -> None:
        """断开所有MCP服务器连接"""
        for server_name, connection in list(self._connections.items()):
            try:
                await connection.disconnect()
            except Exception as e:
                logger.error(
                    "Failed to disconnect from MCP server",
                    server=server_name,
                    error=str(e),
                )
        
        self._connections.clear()
        self._tool_wrappers.clear()
        self._tool_to_server.clear()
    
    def get_tool(self, tool_name: str) -> Optional[MCPToolWrapper]:
        """获取工具包装器"""
        return self._tool_wrappers.get(tool_name)
    
    def get_tools(self) -> List[MCPToolWrapper]:
        """获取所有工具"""
        return list(self._tool_wrappers.values())
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self._tool_wrappers.keys())
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """执行工具"""
        if tool_name in self._tool_wrappers:
            wrapper = self._tool_wrappers[tool_name]
            server_name = wrapper.server_name
            actual_tool_name = wrapper.tool_name
        elif tool_name in self._tool_to_server:
            server_name = self._tool_to_server[tool_name]
            actual_tool_name = tool_name
        else:
            raise MCPToolNotFoundError(tool_name)
        
        if server_name not in self._connections:
            raise MCPError(
                f"MCP server not connected: {server_name}",
                server_name,
            )
        
        connection = self._connections[server_name]
        
        logger.info(
            "Executing MCP tool",
            server=server_name,
            tool=actual_tool_name,
            arguments=arguments,
        )
        
        result = await connection.execute_tool(actual_tool_name, arguments)
        
        logger.debug(
            "MCP tool executed",
            server=server_name,
            tool=actual_tool_name,
            result_length=len(result),
        )
        
        return result
    
    def get_server_tools(self, server_name: str) -> List[MCPToolWrapper]:
        """获取指定服务器的所有工具"""
        return [
            wrapper
            for wrapper in self._tool_wrappers.values()
            if wrapper.server_name == server_name
        ]
    
    def list_servers(self) -> List[str]:
        """列出所有已连接的服务器"""
        return list(self._connections.keys())
    
    def is_server_connected(self, server_name: str) -> bool:
        """检查服务器是否已连接"""
        if server_name in self._connections:
            return self._connections[server_name].is_connected()
        return False
