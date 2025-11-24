import os

from dotenv import load_dotenv
from openai import OpenAI
from openai._types import NotGiven

load_dotenv(override=True)

try:
    key = os.getenv("GPT_key")
    CLIENT = OpenAI(api_key=os.getenv("GPT_key"))
except Exception as e:
    print("Please check the GPT API key")
    print(f"Error: {e}, key: {key}")


def txt2txt(
    input_text="",
    model="gpt-4o",
    system_message="You are an AI assistant that helps people find information.",
    top_p=0,
    temp=0,
):
    """
    use openai to get the summarization of the text
    """
    global CLIENT
    if isinstance(input_text, list):
        input_text = input_text[0]

    if top_p is None:
        top_p = NotGiven
    if temp is None:
        temp = NotGiven

    # response_format={"type": "json_object"},
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message + " Give me the answer as dict only",
            },
            {"role": "user", "content": f"{input_text}"},
        ],
        top_p=top_p,
        temperature=temp,
    )

    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output


def update_con_decon(con_decon_dict, synonyms_dict):
    """
    update the con_decon_dict to make it compatible with sympnonyms option
    """
    # get adjective synonyms
    update_con_decon_dict = con_decon_dict.copy()

    # get all concept adj list, e.g. gun, nude and etc.
    concept_adj_list = [*synonyms_dict.keys()]

    for k, v in con_decon_dict.items():
        for adj in concept_adj_list:
            if adj in k:
                # update con_decon_dict with new key and value
                synon_adj_list = synonyms_dict[adj]
                for synon_adj in synon_adj_list:
                    new_key = k.replace(adj, synon_adj)
                    update_con_decon_dict[new_key] = v
    return update_con_decon_dict
