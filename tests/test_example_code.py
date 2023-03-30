import unittest
import pandas as pd
from unittest.mock import patch
import sys
from example_code import (
    prompt_gpt_4_to_explore_data,
    prompt_gpt_4_to_plot_data,
    get_python_code,
    filter_generated_sequence,
)


class TestApp(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.sample_query = "Show records where A > 1"

    @patch("example_code.openai.ChatCompletion.create")
    def test_prompt_gpt_4_to_explore_data(self, mock_create):
        mock_content = "df[df['A'] > 1]"
        mock_create.return_value = {"choices": [{"message": {"content": mock_content}}]}
        result = prompt_gpt_4_to_explore_data(self.sample_query, self.sample_df)
        self.assertEqual(result.strip(), mock_content)

    @patch("example_code.openai.ChatCompletion.create")
    def test_prompt_gpt_4_to_plot_data(self, mock_create):
        mock_content = "```python\ndef app(df):\n    st.write(df[df['A'] > 1])\n```"
        mock_create.return_value = {"choices": [{"message": {"content": mock_content}}]}
        result = prompt_gpt_4_to_plot_data(self.sample_query, self.sample_df)
        self.assertEqual(result.strip(), mock_content)

    def test_get_python_code(self):
        input_text = "```python\ndef app(df):\n    st.write(df[df['A'] > 1])\n```"
        expected_output = "\ndef app(df):\n    st.write(df[df['A'] > 1])"
        self.assertEqual(get_python_code(input_text), expected_output)

    def test_filter_generated_sequence(self):
        input_text = "```python\ndef app(df):\n    st.write(df[df['A'] > 1])\n```"
        expected_output = "def app(df):\n    st.write(df[df['A'] > 1])"
        self.assertEqual(filter_generated_sequence(input_text), expected_output)
        input_text = "df[df['A'] > 1]"
        self.assertEqual(filter_generated_sequence(input_text), input_text)


if __name__ == "__main__":
    unittest.main()
