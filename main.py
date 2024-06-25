from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_python_agent
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
load_dotenv()

# We are building an agent (ReAct Agent) which can write and execute python code in Python interpretor
# We have an experimental feature called python REPL, it is not in the production langchain as it still needs testing
# This REPL gives llm the ability to write and interpret code in python

def python_agent_router(user_prompt):

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """



    ###########


    ######### python code agent - (Agent - 1)

    # from langchain.agents import (
    #     create_react_agent,
    #     AgentExecutor,
    # )
    #
    # tools = [PythonREPLTool()]
    #
    # base_prompt_template = hub.pull("langchain-ai/react-agent-template")
    #
    # # partial is used to declare variables within the prompt. You can check teh prompt on langchain to see what variables it takes
    # # Same as we created react agent in proj -1 (linkedin/twitter summarizer with agents)
    # prompt = base_prompt_template.partial(instructions=instructions, tools = tools )
    #
    # agent = create_react_agent(ChatOpenAI(temperature=0, model='gpt-4-turbo'), tools=tools, prompt=prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # result = agent_executor.invoke(
    #     input={"input": """ generate and save in the current working directory 15 QR codes that point to www.google.com, you have the qrcode package installed already"""}
    # )

    #alternative python code agent creation - (Agent - 1.1)
    python_agent  = create_python_agent(llm = ChatOpenAI(temperature=0, model='gpt-4'), tool=PythonREPLTool(), verbose=True)
    # The below invoke function is used to invoke the agent manually with the input we want without being routed from routing agent. As we afre using routing agent i commented it
    # python_agent.invoke(
    #     input={
    #         "input": """generate and save in the current working directory 15 QR codes that point to www.google.com, you have the qrcode package installed already"""}
    # )

    ############# csv agent - (Agent - 2) - Only run one agent at a time in the code. If you are using csv agent, then comment python code agent and vice versa. Tools can be chosen by LLM for an agent, but all the agents placed inside the code will run.
    ## we have a langchain builtin function create_csv_agent. This is built on pandas dataframe agent, so no additional tools needed for it
    csv_agent  = create_csv_agent(ChatOpenAI(temperature=0, model='gpt-4'), path="episode_info.csv", verbose=True, allow_dangerous_code=True)
    # The below invoke function is used to invoke the agent manually with the input we want without being routed from routing agent. As we afre using routing agent i commented it
    # csv_agent.invoke(
    #     input={
    #         "input": """Print seasons in ascending order based on the number of episodes in each season"""}
    # )
    ############# csv agent end

    ############## Routing Agent - Helps to choose the right agent based on th user query #############
    # Same as creating an agent with its tools. Here, the tools are nothing but another agents which have to be invoked. And the function for that tools is just their agent invoke method.
    tools = [
        Tool(
            name="Python_Agent",
            func=python_agent.invoke,
            description="""Useful when you need to transform natural language into Python and execute python code, returning the results of the code execution. DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name="CSV_Agent",
            func=csv_agent.invoke,
            description="""Useful when you need to answer questions over episode_info.csv file, takes input the entire question and returns the answer after running pandas calculations"""
        ),

    ]

    base_prompt_template = hub.pull("langchain-ai/react-agent-template")

    prompt = base_prompt_template.partial(instructions="Choose one from the list of tools", tools = tools)

    agent = create_react_agent(ChatOpenAI(temperature=0, model='gpt-4-turbo'), tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke(
        input={"input": user_prompt}
    )
    print(result)

python_agent_router("""Can you write me hello world code?""")