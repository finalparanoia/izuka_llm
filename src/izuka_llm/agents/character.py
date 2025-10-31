import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. 设置 API 密钥 ---
# 为了安全，建议从环境变量读取
# os.environ["OPENAI_API_KEY"] = "sk-..."
# os.environ["TAVILY_API_KEY"] = "tvly-..."

# 如果没有设置环境变量，可以在这里手动设置（不推荐在生产环境中这样做）
try:
    from getpass import getpass

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
    if "TAVILY_API_KEY" not in os.environ:
        os.environ["TAVILY_API_KEY"] = getpass("Enter your Tavily API key: ")
except Exception as e:
    print(f"Error setting API keys: {e}")
    exit()



# --- 3. 定义图的状态 ---
# 状态将在图的节点之间传递
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- 4. 定义模型和工具执行器 ---
# 我们使用 ChatOpenAI 模型
model = ChatOpenAI(model="gpt-4o", temperature=0)
# 将工具绑定到模型上，这样模型就知道如何调用它们
model = model.bind_tools(tools)
# 创建一个工具执行器，用于实际运行工具
tool_executor = ToolExecutor(tools)


# --- 5. 定义图的节点 ---

# 节点1: Agent (负责思考和决定行动)
def agent_node(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


# 节点2: Tools (负责执行工具)
def tools_node(state: AgentState):
    messages = state["messages"]
    # 获取最后一条AI消息，其中应包含工具调用
    last_message = messages[-1]
    # 执行工具调用
    tool_outputs = tool_executor.batch(last_message.tool_calls, return_exceptions=True)

    # 创建工具消息
    tool_messages = []
    for output, tool_call in zip(tool_outputs, last_message.tool_calls):
        if isinstance(output, BaseException):
            tool_messages.append(
                ToolMessage(
                    content=f"Error: {repr(output)}",
                    tool_call_id=tool_call["id"],
                )
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": tool_messages}


# --- 6. 定义图的边（逻辑流程） ---

# 条件边: 决定下一步是调用工具还是结束
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # 如果AI决定调用工具，就进入'tools'节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则，结束流程
    return END


# --- 7. 构建图 ---
# 定义一个新的图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "END": END,
    },
)

# 添加从'tools'节点回到'agent'节点的普通边
# 这样，工具执行完后，AI可以再次思考
workflow.add_edge("tools", "agent")

# 编译图
# MemorySaver 用于在多次运行之间保存状态（可选，但推荐）
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

# --- 8. 运行图 ---
if __name__ == "__main__":
    initial_messages = [HumanMessage(content="旧金山市现任市长的名字是什么？他/她的任期是从哪一年开始的？")]

    # 使用一个线程ID来保存对话历史
    thread = {"configurable": {"thread_id": "1"}}

    print("--- 开始运行 ReAct 智能体 ---")

    # 流式输出，观察每一步
    for event in app.stream({"messages": initial_messages}, thread):
        for node_name, node_output in event.items():
            print(f"--- 节点: {node_name} ---")
            # 打印该节点产生的最后一条消息
            last_message = node_output["messages"][-1]
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    print(f"AI 思考结果: 决定调用工具")
                    for tool_call in last_message.tool_calls:
                        print(f"  - 工具: {tool_call['name']}")
                        print(f"  - 参数: {tool_call['args']}")
                else:
                    print(f"AI 最终回答: {last_message.content}")
            elif isinstance(last_message, ToolMessage):
                print(f"工具执行结果: {last_message.content[:200]}...")  # 只打印前200个字符
        print("-" * 20)

    print("\n--- 最终对话历史 ---")
    final_state = app.get_state(thread)
    for message in final_state.values["messages"]:
        message.pretty_print()

