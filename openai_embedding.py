import json
import re
from openai import OpenAI
import pandas as pd
import tiktoken
import os
import dspy





if __name__ == '__main__':
    text_file_loc = 'clusters csv'
    text_col_name = 'text'

    api_key = ('API-KEY-HERE')
    client = OpenAI(api_key=api_key)


