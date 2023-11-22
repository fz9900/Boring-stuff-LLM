# /home/project/knowledgebase.py
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from dotenv import load_dotenv
import os

load_dotenv()

BEGIN_USER_INSTRUCTIONS = "[INST]"
END_USER_INSTRUCTIONS = "[/INST]"
BEGIN_SYSTEM_INSTRUCTIONS = "<<SYS>>"
END_SYSTEM_INSTRUCTIONS = "<</SYS>>"

def prompt_contains_llama_tokens(prompt):
    llama_tokens = [BEGIN_USER_INSTRUCTIONS, END_USER_INSTRUCTIONS, BEGIN_SYSTEM_INSTRUCTIONS, END_SYSTEM_INSTRUCTIONS]
    return any(token in prompt for token in llama_tokens)

def format_prompt(system_instructions, user_instructions, response_prefix=None):
    return f"{BEGIN_SYSTEM_INSTRUCTIONS}\n{system_instructions.strip()}\n{END_SYSTEM_INSTRUCTIONS}\n\n{BEGIN_USER_INSTRUCTIONS}{user_instructions.strip()}{END_USER_INSTRUCTIONS}\n{response_prefix if response_prefix else ''}"


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

with open("call-center-llm-samples/data/knowledgebase.md", "r") as kb:
    kb_text = kb.read()

SYSTEM_PROMPT = f"""
Answer questions as a call center AI assistant. 
Try to answer as succinctly as possible. 
Do not provide generic advice. You only use the below knowledgebase to answer questions. 
Only answer with information in the knowledgebase. 
If a question is not answered in the knowledgebase, respond that you do not know.

{kb_text}
"""

USER_PROMPT = "How can I activate waveguard?"

if prompt_contains_llama_tokens(USER_PROMPT):
    print("potentially malicious input detected")
else:
    formatted_prompt = format_prompt(SYSTEM_PROMPT, USER_PROMPT)
    generated_response = model.generate(prompt=formatted_prompt)
    print(generated_response['results'][0]['generated_text'].strip())