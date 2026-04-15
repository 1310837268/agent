"""
MCP Tools集成模块（增强版）

修复了资源管理问题：
1. 使用正确的上下文管理器模式
2. 连接失败时的资源兜底释放
3. 异常安全的资源清理
4. 集成资源追踪器
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from agent_system.core.config import MCPConfig
from agent_system.core.exceptions import MCPError, MCPToolNotFoundError
from agent_system.core.logging import get_logger
from agent_system.core.resource_manager import (
    ManagedResource,
    ResourceState,
    ResourceTracker,
)

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


class MCPServerConnection(ManagedResource):
    """
    MCP服务器连接（受管资源）
    
    继承自ManagedResource，确保资源正确管理：
    1. 连接时获取资源
    2. 断开时释放资源
    3. 异常时兜底释放
    4. 资源泄漏检测
    """
    
    def __init__(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name=f"mcp_server:{server_name}")
        
        self.server_name = server_name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._session: Optional[Any] = None
        self._client: Optional[Any] = None
        self._client_context: Optional[Any] = None
        self._session_context: Optional[Any] = None
        self._tools: List[Dict[str, Any]] = []
    
    async def _acquire(self) -> None:
        """
        获取资源（连接到MCP服务器）
        
        使用正确的上下文管理器模式，确保异常安全
        """
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP is not installed. Install with: pip install mcp mcp-client"
            )
        
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        
        client = None
        session = None
        client_context = None
        session_context = None
        
        try:
            logger.info(
                "Connecting to MCP server",
                server=self.server_name,
                command=self.command,
            )
            
            client_context = stdio_client(server_params)
            read, write = await client_context.__aenter__()
            client = client_context
            
            session_context = ClientSession(read, write)
            await session_context.__aenter__()
            session = session_context
            
            await session.initialize()
            
            tools_response = await session.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema:": tool.inputSchema.model_dump() 
                    if hasattr(tool.inputSchema, 'model_dump') 
                    else dict(tool.inputSchema),
                }
                for tool in tools_response.tools
            ]
            
            self._client = client
            self._session = session
            self._client_context = client_context
            self._session_context = session_context
            
            logger.info(
                "Connected to MCP server successfully",
                server=self.server_name,
                tools=[t["name"] for t in self._tools],
            )
            
        except Exception as e:
            logger.error(
                "Failed to connect to MCP server",
                server=self.server_name,
                error=str(e),
            )
            
            await self._safe_cleanup(
                client_context,
                session_context,
                client,
                session,
            )
            
            raise
    
    async def _safe_cleanup(
        self,
        client_context: Optional[Any] = None,
        session_context: Optional[Any] = None,
        client: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> None:
        """
        安全清理资源
        
        确保即使在异常情况下也能正确释放资源
        """
        cleanup_errors = []
        
        if session_context is not None:
            try:
                await session_context.__aexit__(None, None, None)
                logger.debug("Session context cleaned up", server=self.server_name)
            except Exception as e:
                cleanup_errors.append(f"session_context: {str(e)}")
                logger.warning(
                    "Failed to cleanup session context",
                    server=self.server_name,
                    error=str(e),
                )
        
        if client_context is not None:
            try:
                await client_context.__aexit__(None, None, None)
                logger.debug("Client context cleaned up", server=self.server_name)
            except Exception as e:
                cleanup_errors.append(f"client_context: {str(e)}")
                logger.warning(
                    "Failed to cleanup client context",
                    server=self.server_name,
                    error=str(e),
                )
        
        if cleanup_errors:
            logger.warning(
                "Some cleanup errors occurred",
                server=self.server_name,
                errors=cleanup_errors,
            )
    
    async def _release(self) -> None:
        """
        释放资源（断开连接但不关闭）
        
        对于MCP连接，释放和关闭是相同的操作
        """
        await self._close()
    
    async def _close(self) -> None:
        """
        关闭资源（断开MCP服务器连接）
        
        确保所有资源都被正确释放
        """
        logger.info("Disconnecting from MCP server", server=self.server_name)
        
        await self._safe_cleanup(
            self._session_context,
            self._client_context,
            self._client,
            self._session,
        )
        
        self._session = None
        self._client = None
        self._session_context = None
        self._client_context = None
        self._tools = []
        
        logger.info("Disconnected from MCP server", server=self.server_name)
    
    @asynccontextmanager
    async def managed_connection(self) -> AsyncIterator["MCPServerConnection"]:
        """
        受管连接上下文管理器
        
        Usage:
            async with connection.managed_connection() as conn:
                # 使用连接
                pass
        """
        try:
            await self.acquire()
            yield self
        except Exception as e:
            logger.error(
                "Error in managed MCP connection",
                server=self.server_name,
                error=str(e),
            )
            raise
        finally:
            if self.is_acquired:
                await self.release()
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """执行MCP工具"""
        if not self._session or not self.is_acquired:
            raise MCPError(
                f"Not connected to MCP server: {self.server_name}",
                self.server_name,
            )
        
        logger.info(
            "Executing MCP tool",
            server=self.server_name,
            tool=tool_name,
            arguments=arguments,
        )
        
        try:
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
            
            result_str = "\n".join(content_parts)
            
            logger.debug(
                "MCP tool executed",
                server=self.server_name,
                tool=tool_name,
                result_length=len(result_str),
            )
            
            return result_str
            
        except Exception as e:
            logger.error(
                "Failed to execute MCP tool",
                server=self.server_name,
                tool=tool_name,
                error=str(e),
            )
            raise MCPError(
                f"Failed to execute tool '{tool_name}': {str(e)}",
                self.server_name,
            ) from e
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """获取所有工具"""
        return self._tools
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.is_acquired and self._session is not None


class MCPManager:
    """
    MCP管理器（增强版）
    
    管理多个MCP服务器连接，提供：
    1. 异常安全的连接管理
    2. 资源兜底释放
    3. 优雅关闭机制
    """
    
    def __init__(self, config: MCPConfig) -> None:
        self.config = config
        self._connections: Dict[str, MCPServerConnection] = {}
        self._tool_wrappers: Dict[str, MCPToolWrapper] = {}
        self._tool_to_server: Dict[str, str] = {}
        self._connected = False
        
        ResourceTracker.register(self)
    
    async def connect_all(self) -> None:
        """
        连接所有配置的MCP服务器
        
        异常安全：即使某些服务器连接失败，也会确保已连接的资源正确管理
        """
        if self._connected:
            logger.warning("MCPManager already connected")
            return
        
        connected_servers = []
        
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
                await connection.acquire()
                self._connections[server_name] = connection
                connected_servers.append(server_name)
                
                for tool in connection.get_tools():
                    wrapper = MCPToolWrapper(
                        server_name=server_name,
                        tool_name=tool["name"],
                        tool_description=tool["description"],
                        input_schema=tool.get("input_schema", {}),
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
                
                try:
                    if connection.is_acquired:
                        await connection.close()
                except Exception as cleanup_error:
                    logger.error(
                        "Failed to cleanup failed connection",
                        server=server_name,
                        error=str(cleanup_error),
                    )
        
        self._connected = True
        
        if connected_servers:
            logger.info(
                "MCPManager connected to servers",
                servers=connected_servers,
            )
        else:
            logger.warning("No MCP servers connected successfully")
    
    async def disconnect_all(self) -> None:
        """
        断开所有MCP服务器连接
        
        确保所有资源都被正确释放，即使某些断开操作失败
        """
        if not self._connected:
            return
        
        logger.info("Disconnecting all MCP servers")
        
        disconnect_errors = []
        
        for server_name, connection in list(self._connections.items()):
            try:
                await connection.close()
                logger.info("MCP server disconnected", server=server_name)
            except Exception as e:
                disconnect_errors.append(f"{server_name}: {str(e)}")
                logger.error(
                    "Failed to disconnect MCP server",
                    server=server_name,
                    error=str(e),
                )
        
        self._connections.clear()
        self._tool_wrappers.clear()
        self._tool_to_server.clear()
        self._connected = False
        
        if disconnect_errors:
            logger.warning(
                "Some MCP servers failed to disconnect",
                errors=disconnect_errors,
            )
        else:
            logger.info("All MCP servers disconnected successfully")
    
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
        
        return await connection.execute_tool(actual_tool_name, arguments)
    
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
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
    
    async def __aenter__(self) -> "MCPManager":
        """异步上下文管理器入口"""
        await self.connect_all()
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """异步上下文管理器出口"""
        await self.disconnect_all()
    
    def __del__(self) -> None:
        """析构函数 - 检测资源泄漏"""
        if self._connected:
            logger.warning(
                "MCPManager was not properly disconnected",
                servers=list(self._connections.keys()),
            )
