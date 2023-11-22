# /home/project/evaluate.py
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from dotenv import load_dotenv
import os
import json

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
    GenParams.MAX_NEW_TOKENS: 900, # 900 is the max number of output tokens for LLAMA_2_70B_CHAT for the moment
    "stop_sequences": ["}"] # <--- NOTE: for this example we've added a new stop sequence
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



# Feel free to experiment with these
SYSTEM_PROMPT = """
    You are an AI assistant at an ISP call center specializing in evaluating call center agent performance based on call transcripts. 
    You are provided a transcript and will return a JSON response.
    The structure of the response is: {"identityConfirmed": boolean, "reassuranceStatement": boolean, "reasonForCallResolved": boolean, "advisedOfSurvey": boolean}
    "identityConfirmed": true if the agent was able to confirm the customer's identity by asking their full name and PIN
    "reassuranceStatement": true if the agent provided a statement of reassurance after the customer explained the problem (e.g. I can help you resolve that today)"
    "reasonForCallResolved": true if the agent asked if the reason for the call was resolved
    "advisedOfSurvey": true if the agent advised the customer that they will receive a survey about today's call and explained that the survey is for the agent's performance and not the quality of internet service
    """
ANSWER_PREFIX = "JSON response: "

PATH_TO_TRANSCRIPTS = "call-center-llm-samples/data/transcripts"
for filename in sorted(os.listdir(PATH_TO_TRANSCRIPTS)):
    with open(f"{PATH_TO_TRANSCRIPTS}/{filename}", "r") as transcript:
        transcript_text = transcript.read()
        if prompt_contains_llama_tokens(transcript): # seems unlikely, but just to be safe...
            print("potentially malicious input detected")
        else:
            formatted_prompt = format_prompt(SYSTEM_PROMPT, transcript_text, ANSWER_PREFIX)
            generated_response = model.generate(prompt=formatted_prompt)
            response_text = generated_response['results'][0]['generated_text'].strip()

            print(f"{filename}:")
            try:
                parsed_evaluation = json.loads(response_text)
                assert "identityConfirmed" in parsed_evaluation
                assert "reassuranceStatement" in parsed_evaluation
                assert "reasonForCallResolved" in parsed_evaluation
                assert "advisedOfSurvey" in parsed_evaluation
            except json.decoder.JSONDecodeError:
                print("Your response was not proper JSON.")
                print("Your response:")
                print(response_text)
            except AssertionError as e:
                print("One of the expected keys was not found.")
                print("Double check that you don't have a typo in your prompt")
                print(e)

            print(generated_response['results'][0]['generated_text'].strip())