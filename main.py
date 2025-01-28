import argparse
import json
import os

from langchain import hub
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDatabaseTool,
)
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine


def run_sql_agent(prompt: HumanMessage):
    llm = ChatOllama(model="llama3.1", temperature=0)
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_message = SystemMessage(
        prompt_template.format(dialect="SQLServer", top_k=5)
        + "\n All tables will be captialized and plural."
    )

    db = init_sql_database()
    info_sql_database_tool_description = """
	    Input to this tool is a comma separated list of tables, output is the schema and sample rows for those tables.
	    Be sure that the tables actually exist by calling list_tables_sql_db first!
	    Example Input: table1, table2, table3"""

    tools = [
        QuerySQLCheckerTool(db=db, llm=llm),
        InfoSQLDatabaseTool(db=db, description=info_sql_database_tool_description),
        ListSQLDatabaseTool(db=db),
        QuerySQLDatabaseTool(db=db),
    ]

    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

    events = agent_executor.stream(
        {"messages": prompt},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()


def init_sql_database():
    alchemyStr = os.environ.get("DB_ALCHEMY_CONNECTION_STRING")
    if alchemyStr is None:
        raise ValueError("DB_ALCHEMY_CONNECTION_STRING is not set in the environment.")
    engine = create_engine(alchemyStr)
    db = SQLDatabase(engine)
    print("Info: Database connection established.")
    return db


def _set_env(key: str, value: str):
    if key not in os.environ:
        os.environ[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    # Load the secrets from the JSON file and set them as environment variables
    with open("secrets.json", "r") as f:
        secrets = json.load(f)
        for key, value in secrets.items():
            _set_env(key, value)

    prompt = HumanMessage(args.prompt)
    run_sql_agent(prompt)


if __name__ == "__main__":
    main()
