import argparse
import json
import os

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine


# Builds and runs the SQL agent
def run_sql_agent(prompt):
    # llm = ChatOpenAI(model="gpt-4o-mini")
    dbEngine = get_engine_for_sql_server_db()
    db = SQLDatabase(dbEngine)
    print(db.table_info)


#    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#    tools = toolkit.get_tools()

#    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")
#    assert len(prompt_template.messages) == 1

#    system_message = """System: You are an agent designed to interact with a SQL database filled with various account and tax data. Your name is ShankDuck.
# Given an input question, create a syntactically correct SQLServer query to run, then look at the results of the query and return the answer.
# You can order the results by a relevant column to return the most interesting examples in the database.
# Never query for all the columns from a specific table, only ask for the relevant columns given the question.
# You have access to tools for interacting with the database.
# Only use the below tools. Only use the information returned by the below tools to construct your final answer.
# You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.


# To start you should ALWAYS look at the tables in the database to see what you can query.
# Do NOT skip this step.
# Then you should query the schema of the most relevant tables."""
#


# Build SQLServer engine
def get_engine_for_sql_server_db():
    alchemyStr = os.environ.get("DB_ALCHEMY_CONNECTION_STRING")
    connection_string = f"mssql+pyodbc://{alchemyStr}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    return create_engine(connection_string)


# Assign environment variables
def _set_env(key: str, value):
    if key not in os.environ:
        os.environ[key] = value


## Parse the command line arguments and run the agent
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
