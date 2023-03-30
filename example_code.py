import openai
import os
import pandas as pd
import streamlit as st
import pandas as pd
import seaborn as sns

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
            {"role": "user", "content": f"Here's an example from a dataset: {sample}"},
            {
                "role": "user",
                "content": f"Convert this query into a seaborn plot.  Write a one-line python command: {input_message}",
            },
            {"role": "assistant", "content": "plot ="},
        ],
    )
    generated_text = response["choices"][0]["message"]["content"].strip()
    print(f"Generated response: {generated_text}")
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

    query = st.text_input("Enter query:")

    text = prompt_gpt_4_to_explore_data(query, df)
    print(text)
    text = filter_generated_sequence(text)

    new_df = eval(text)

    # Display the filtered dataframe
    st.write(new_df)

    new_query = st.text_input("Enter query 2:")

    text = prompt_gpt_4_to_plot_data(new_query, new_df)
    print(text)
    text = filter_generated_sequence(text)

    plot = eval(text)

    st.pyplot(plot)


def filter_generated_sequence(text):
    if "```python" in text:
        text = get_python_code(text)
        print(text)
    if " = " in text:
        text = text.split(" = ")[1]
        print(text)
    return text


if __name__ == "__main__":
    app()
