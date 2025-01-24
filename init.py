import argparse


def run_sql_agent(prompt):
    print(prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    run_sql_agent(args.prompt)


if __name__ == "__main__":
    main()
