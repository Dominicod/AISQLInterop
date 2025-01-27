import argparse
import json
import os

from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine


def run_sql_agent(prompt: str):
    llm = ChatOllama(model="mistral")
    db = init_sql_database()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1
    system_message = prompt_template.format(dialect="SQLServer", top_k=5)

    agent_executor = create_react_agent(
        llm, toolkit.get_tools(), state_modifier=system_message
    )
    events = agent_executor.stream(
        {"messages": [("user", prompt)]},
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

    run_sql_agent(args.prompt)


if __name__ == "__main__":
    main()
