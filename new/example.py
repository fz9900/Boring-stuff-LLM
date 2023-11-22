# /home/project/example.py
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from dotenv import load_dotenv
import os

# loads our api credentials from .env into our environment
load_dotenv()

# Llama system instruction tokens
BEGIN_SYSTEM_INSTRUCTIONS = "<<SYS>>"
END_SYSTEM_INSTRUCTIONS = "<</SYS>>"

# Llama user prompt/instructions
BEGIN_USER_PROMPT = "[INST]"
END_USER_PROMPT = "[/INST]"

def prompt_contains_llama_tokens(prompt):
    """returns true if the prompt contains llama tokens (which may be malicious!)"""
    llama_tokens = [BEGIN_USER_PROMPT, END_USER_PROMPT, BEGIN_SYSTEM_INSTRUCTIONS, END_SYSTEM_INSTRUCTIONS]
    return any(token in prompt for token in llama_tokens)

def format_prompt(system_instructions, user_prompt, response_prefix=None):
    return f"{BEGIN_SYSTEM_INSTRUCTIONS}\n{system_instructions.strip()}\n{END_SYSTEM_INSTRUCTIONS}\n\n{BEGIN_USER_PROMPT}{user_prompt.strip()}{END_USER_PROMPT}\n{response_prefix if response_prefix else ''}"


generate_params = {
    GenParams.MAX_NEW_TOKENS: 900 # 900 is the max number of output tokens for LLAMA_2_70B_CHAT for the moment
}

model = Model(
    model_id=ModelTypes.LLAMA_2_70B_CHAT,
    params=generate_params,
    credentials={
        "apikey": os.environ["IBMCLOUD_API_KEY"],
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id=os.environ["PROJECT_ID"]
    )


##############################################################
# Please have fun experimenting with the three variables below
##############################################################
# system_prompt controls the behaviour of the LLM
system_prompt = "You are an AI assistant. Answer all prompts as succinctly as possible, without unnecessary commentary."

# user_prompt is the
user_prompt = "What year was Las Vegas founded?"

# response prefix lets you write the beginning of the LLM's response
# to force the LLM answer to take a certain form
response_prefix = "Las Vegas was founded in"


if prompt_contains_llama_tokens(user_prompt):
    print("potentially malicious input detected")
else:
    formatted_prompt = format_prompt(system_prompt, user_prompt, response_prefix)
    completed_response = model.generate(prompt=formatted_prompt)['results'][0]['generated_text'].strip()
    composed_response = f"{response_prefix} {completed_response}"
    print(composed_response)