import openai
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

openai.api_key = os.getenv("OPENAI_API_KEY")


def prompt_gpt_4_to_explore_data(input_message, df):
    sample = df.sample(3)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a expert data analyst, with fantastic pandas skills.",
            },
            {"role": "user", "content": f"Here's an example from a dataset: {sample}"},
            {
                "role": "user",
                "content": f"Convert this query into a one-liner pandas command that returns a dataframe.  Don't perform in-place: {input_message}",
            },
            {"role": "assistant", "content": "df ="},
        ],
    )
    generated_text = response["choices"][0]["message"]["content"].strip()
    print(f"Generated response: {generated_text}")
    return generated_text


def prompt_gpt_4_to_plot_data(input_message, df):

    sample = df.sample(3)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a expert data analyst, with fantastic pandas skills.",
            },
            {
                "role": "user",
                "content": f"Here's an example from a dataframe: {sample}",
            },
            {
                "role": "user",
                "content": f"Create a function `def app(df)` to represent this query as streamlit code. The dataframe is already a global variable named df: {input_message}",
            },
        ],
        temperature=0.9
    )
    generated_text = response["choices"][0]["message"]["content"].strip()
    print(f"Generated response: {generated_text}")

    generated_text = filter_generated_sequence(generated_text)

    if generated_text.count("app(df)") < 2:
        generated_text += "\n\napp(df)"

    return generated_text


def get_python_code(generated_text):

    # extract the text between the ```python blocks
    python_code = generated_text.split("```python")[1].split("```")[0]
    return python_code


df = pd.read_json("food-enforcement.json")

# Define the Streamlit app
def app():
    st.title("My Dataframe")
    st.write(df)

    new_query = st.text_input("Enter query to turn into streamlit code:")
    text = prompt_gpt_4_to_plot_data(new_query, df)
    print(text)
    text = filter_generated_sequence(text)
    exec(text)


def filter_generated_sequence(text):
    if "```python" in text:
        text = get_python_code(text)
        text = text.strip(" \n")
        print(text)
    return text


if __name__ == "__main__":
    app()
