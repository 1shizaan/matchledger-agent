import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from app.utils.query_parser import parse_natural_date_range # This import is correct and needed

from dotenv import load_dotenv

load_dotenv()

def run_chat_agent(ledger_df: pd.DataFrame, bank_df: pd.DataFrame, query: str):
    combined_df = pd.concat([
        ledger_df.assign(source="Ledger"),
        bank_df.assign(source="Bank")
    ])

    # --- START OF NEW CODE FOR DATE FILTERING ---
    start, end = parse_natural_date_range(query)
    if start and end:
        # It's good practice to ensure the 'date' column exists and is in datetime format
        # before trying to convert it and filter.
        if 'date' in combined_df.columns:
            # Convert 'date' column to datetime, handling potential errors
            # 'coerce' will turn unparseable dates into NaT (Not a Time)
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')

            # Drop rows where date conversion failed (NaT values) if you don't want them
            combined_df = combined_df.dropna(subset=['date'])

            # Filter the DataFrame based on the parsed date range
            combined_df = combined_df[
                (combined_df['date'].dt.date >= start) &
                (combined_df['date'].dt.date <= end)
            ]
            print(f"DEBUG: Filtered data for date range: {start} to {end}. Remaining rows: {len(combined_df)}")
        else:
            print("Warning: 'date' column not found in combined_df. Skipping date range filtering.")
    # --- END OF NEW CODE FOR DATE FILTERING ---

    # The agent will now operate on the potentially filtered combined_df
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        combined_df,
        verbose=True, # Keeping this True for better debugging of agent's thoughts
        allow_dangerous_code=True,
        max_iterations=15
    )

    result = agent.invoke({"input": query})
    return result["output"]