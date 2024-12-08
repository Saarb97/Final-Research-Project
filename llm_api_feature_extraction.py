import json
import re
from openai import OpenAI
import pandas as pd
import tiktoken
import os
import dspy

def _count_cluster_files(clusters_files_loc: str):
    # Match files with the pattern "<cluster>_data.csv"
    cluster_files = [
        f for f in os.listdir(clusters_files_loc)
        if os.path.isfile(os.path.join(clusters_files_loc, f)) and f.endswith("_data.csv")
    ]
    return len(cluster_files)


def _count_tokens(prompt: str, model: str="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))


def _create_cluster_prompt(prompt: str, text_file_loc: str, text_col_name: str, themes: list = None):
    data = pd.read_csv(text_file_loc)
    cluster_text = data[text_col_name]

    # Create base full prompt
    full_prompt = prompt

    # If themes are provided, add them to the prompt
    if themes:
        themes_list_one_per_line = '\n'.join(themes)
        full_prompt += f"\nThemes:\n{themes_list_one_per_line}"

    # Add cluster texts to the prompt
    full_prompt += f"\nCluster texts:\n" + '\n'.join(cluster_text.astype(str))

    return full_prompt


def _get_subthemes(client: OpenAI, full_prompt: str, model: str="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,  # Replace with your model
        messages=[
            {
                "role": "system",
                "content": "Your purpose is to analyze all the texts in the prompt and return subthemes and not high level themes."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "twenty_five_subthemes_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "twenty_five_subthemes": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["twenty_five_subthemes"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        store=True
    )

    # Parse the JSON content from the response
    content = response.choices[0].message.content
    print(f'full response:\n {response}')
    parsed_content = json.loads(content)  # Convert JSON string to Python dictionary

    # Extract the subthemes list
    subthemes = parsed_content.get("twenty_five_subthemes", [])
    print(f'subthemes amount received: {len(subthemes)}')
    return subthemes


def llm_feature_extraction_for_cluster_csv(client: OpenAI, text_file_loc, text_col_name, model="gpt-4o-mini"):

    first_stage_instruction = (
        "Analyze this series of stories and questions to identify five main recurring themes. "
        "These themes should reflect patterns in characters' actions, traits, and dynamics, capturing both explicit and subtle ideas across the scenarios. "
        "After identifying the five main themes, expand on each theme by identifying five specific and coherent subthemes that provide more detail. "
        "Ensure each theme has exactly five subthemes. Do not skip or summarize steps.\n\n"
        "Present the output in this exact structured format:\n"
        "- Theme 1\n"
        "  - Subtheme 1.1\n"
        "  - Subtheme 1.2\n"
        "  - Subtheme 1.3\n"
        "  - Subtheme 1.4\n"
        "  - Subtheme 1.5\n"
        "- Theme 2\n"
        "  - Subtheme 2.1\n"
        "  - Subtheme 2.2\n"
        "  - Subtheme 2.3\n"
        "  - Subtheme 2.4\n"
        "  - Subtheme 2.5\n"
        "... \n\n"
        "Ensure the final output includes exactly five themes, each with five subthemes, formatted as described above. "
        "This dataset is for testing ML model fairness, so avoid general descriptions or mentions of individuals."
    )

    # If the LLM returned only 5 main themes without going deeper into the sub themes, a second prompt is sent.
    second_stage_instruction = (
        "The following analysis has already identified five main recurring themes. "
        "Your task is to expand on each theme by identifying exactly five specific and coherent subthemes that provide more detail. "
        "These subthemes should reflect deeper patterns, situations, or dynamics associated with each main theme, capturing both explicit and subtle ideas. "
        "Avoid general descriptions or overly broad categories; focus on concrete and precise subthemes related to the data provided.\n\n"
        "Present the output in this exact structured format:\n"
        "- Theme 1\n"
        "  - Subtheme 1.1\n"
        "  - Subtheme 1.2\n"
        "  - Subtheme 1.3\n"
        "  - Subtheme 1.4\n"
        "  - Subtheme 1.5\n"
        "- Theme 2\n"
        "  - Subtheme 2.1\n"
        "  - Subtheme 2.2\n"
        "  - Subtheme 2.3\n"
        "  - Subtheme 2.4\n"
        "  - Subtheme 2.5\n"
        "... \n\n"
        "Ensure the output includes subthemes for all five themes, with exactly five subthemes per theme. "
        "This dataset is for testing ML model fairness, so avoid general descriptions or mentions of individuals."
    )

    first_full_prompt = _create_cluster_prompt(first_stage_instruction, text_file_loc, 'text')
    token_count = _count_tokens(first_full_prompt, model)

    TOKEN_LIMIT = 128_000
    if token_count > TOKEN_LIMIT:
        print(f"Cluster at {text_file_loc} is too large for {model} context window of {TOKEN_LIMIT} "
              f"with {token_count} tokens.")


    repeats = 1
    sub_themes = _get_subthemes(client, first_full_prompt)
    subthemes_len = len(sub_themes)

    # Another option is to look for the "steps" option and define steps to follow - get themes -> get subthemes
    while subthemes_len < 25 and repeats <= 5:
        # In this case Only the main themes were received -> extracting sub-themes.
        if len(sub_themes) == 5:
            second_full_prompt = _create_cluster_prompt(second_stage_instruction, text_file_loc, 'text')
            sub_themes = _get_subthemes(client, second_full_prompt, model)
            subthemes_len = len(sub_themes)
            repeats += 1

        # when atleast 25 sub-themes are received it is an acceptable result.
        elif len(sub_themes) >= 25:
            print(f'{len(sub_themes)} themes were given: \n {sub_themes}')
            subthemes_len = len(sub_themes)

        else:
            sub_themes = _get_subthemes(client, first_full_prompt)
            subthemes_len = len(sub_themes)
            repeats += 1

    return sub_themes


def llm_feature_extraction_for_clusters_folder(client, clusters_files_loc: str, text_col_name: str,
                                               model: str = "gpt-4o-mini") -> pd.DataFrame:

    num_of_clusters = _count_cluster_files(clusters_files_loc)
    llm_features_pd = pd.DataFrame()

    # Loop through all clusters from 0_data.csv to (num_of_clusters - 1)_data.csv
    for i in range(num_of_clusters):
        cluster_file = os.path.join(clusters_files_loc, f"{i}_data.csv")  # Construct file name
        try:
            features = llm_feature_extraction_for_cluster_csv(client, cluster_file, text_col_name, model)
            # llm_features_pd[f"{i}"] = features
            temp_df = pd.DataFrame({f"{i}": features})
            # Concatenate with the main DataFrame, aligning indexes and allowing for NaN values
            llm_features_pd = pd.concat([llm_features_pd, temp_df], axis=1)

        except Exception as e:
            raise(f"Error processing cluster {i}: {e}")

    if len(llm_features_pd.columns) == 0:
        raise Exception(f"No features were generated for all clusters.")
    elif num_of_clusters != len(llm_features_pd.columns):
        print(f"generated features for {len(llm_features_pd.columns)} clusters out of {num_of_clusters} clusters.")
    else:
        print(f"generated features successfully for {num_of_clusters} clusters.")
    return llm_features_pd

if __name__ == '__main__':
    text_file_loc = 'clusters csv'
    text_col_name = 'text'

    api_key = "INSERT API KEY"
    client = OpenAI(api_key=api_key)

    # features = llm_feature_extraction_for_cluster_csv(client, text_file_loc, text_col_name)
    # print(features)
    features_pd = llm_feature_extraction_for_clusters_folder(client, text_file_loc, text_col_name, model="gpt-4o-mini")
    ai_features_file_name = os.path.join(text_file_loc, 'ai_features_df.csv')
    features_pd.to_csv(ai_features_file_name, index=False)
