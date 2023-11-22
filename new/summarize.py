# /home/project/summarize.py
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

def format_prompt(system_instructions, user_prompt, response_prefix=None):
    return f"{BEGIN_SYSTEM_INSTRUCTIONS}\n{system_instructions.strip()}\n{END_SYSTEM_INSTRUCTIONS}\n\n{BEGIN_USER_INSTRUCTIONS}{user_prompt.strip()}{END_USER_INSTRUCTIONS}\n{response_prefix if response_prefix else ''}"


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


# system_prompt controls the behaviour of the LLM
system_prompt = "Agent: Good afternoon, thank you for reaching out to our support center. How can I assist you today?"

"Customer: Hi, my internet has been really slow for the last few hours and its quite frustrating."

"Agent: Can I have your full name, please?"




# response prefix lets you write the beginning of
# the response on the AI's behalf to force the answer to take a certain form
response_prefix = "" # TODO: optionally set a response prefix

PATH_TO_TRANSCRIPTS = "call-center-llm-samples/data/transcripts"
for filename in sorted(os.listdir(PATH_TO_TRANSCRIPTS)):
    with open(f"{PATH_TO_TRANSCRIPTS}/{filename}", "r") as transcript:
        user_prompt = transcript.read()

        if prompt_contains_llama_tokens(user_prompt): # seems unlikely, but just to be safe...
            print("potentially malicious input detected")
        else:
            formatted_prompt = format_prompt(system_prompt, user_prompt, response_prefix)
            completed_response = model.generate(prompt=formatted_prompt)['results'][0]['generated_text'].strip()
            composed_response = f"{completed_response}"
            print(f"summary of transcript {filename}:")
            print(composed_response)