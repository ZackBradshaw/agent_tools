from transformers import AutoTokenizer
from botocore.exceptions import NoCredentialsError
import tokenize
import requests
import os
import time
from functools import partial
from pathlib import Path
from threading import Lock
import warnings
import json
from vllm import LLM 

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')

import matplotlib
import gradio as gr
from tools_controller import MTQuestionAnswerer, load_valid_tools
from singletool import STQuestionAnswerer
from langchain.schema import AgentFinish
import requests
from tool_server import run_tool_server
from threading import Thread
from multiprocessing import Process
import time
from langchain.llms import VLLM
import yaml

matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

tool_server_flag = False

def start_tool_server():
    # server = Thread(target=run_tool_server)
    server = Process(target=run_tool_server)
    server.start()
    global tool_server_flag
    tool_server_flag = True


DEFAULTMODEL = "ChatGPT"  # "GPT-3.5"

# Read the model/ directory and get the list of models
available_models = ["ChatGPT", "GPT-3.5"]

tools_mappings = {
    "klarna": "https://www.klarna.com/",
    "weather": "http://127.0.0.1:8079/tools/weather/",
    # "database": "http://127.0.0.1:8079/tools/database/",
    # "db_diag": "http://127.0.0.1:8079/tools/db_diag/",
    "chemical-prop": "http://127.0.0.1:8079/tools/chemical-prop/",
    "douban-film": "http://127.0.0.1:8079/tools/douban-film/",
    "wikipedia": "http://127.0.0.1:8079/tools/wikipedia/",
    # "wikidata": "http://127.0.0.1:8079/tools/kg/wikidata/",
    "wolframalpha": "http://127.0.0.1:8079/tools/wolframalpha/",
    "bing_search": "http://127.0.0.1:8079/tools/bing_search/",
    "office-ppt": "http://127.0.0.1:8079/tools/office-ppt/",
    "stock": "http://127.0.0.1:8079/tools/stock/",
    "bing_map": "http://127.0.0.1:8079/tools/map.bing_map/",
    # "baidu_map": "http://127.0.0.1:8079/tools/map/baidu_map/",
    "zillow": "http://127.0.0.1:8079/tools/zillow/",
    "airbnb": "http://127.0.0.1:8079/tools/airbnb/",
    "job_search": "http://127.0.0.1:8079/tools/job_search/",
    # "baidu-translation": "http://127.0.0.1:8079/tools/translation/baidu-translation/",
    # "nllb-translation": "http://127.0.0.1:8079/tools/translation/nllb-translation/",
    "tutorial": "http://127.0.0.1:8079/tools/tutorial/",
    "file_operation": "http://127.0.0.1:8079/tools/file_operation/",
    "meta_analysis": "http://127.0.0.1:8079/tools/meta_analysis/",
    "code_interpreter": "http://127.0.0.1:8079/tools/code_interpreter/",
    "arxiv": "http://127.0.0.1:8079/tools/arxiv/",
    "google_places": "http://127.0.0.1:8079/tools/google_places/",
    "google_serper": "http://127.0.0.1:8079/tools/google_serper/",
    "google_scholar": "http://127.0.0.1:8079/tools/google_scholar/",
    "python": "http://127.0.0.1:8079/tools/python/",
    "sceneXplain": "http://127.0.0.1:8079/tools/sceneXplain/",
    "shell": "http://127.0.0.1:8079/tools/shell/",
    "image_generation": "http://127.0.0.1:8079/tools/image_generation/",
    "hugging_tools": "http://127.0.0.1:8079/tools/hugging_tools/",
    "gradio_tools": "http://127.0.0.1:8079/tools/gradio_tools/",
    "travel": "http://127.0.0.1:8079/tools/travel",
    "walmart": "http://127.0.0.1:8079/tools/walmart",
}

# data = json.load(open('sourcery-engine/tools/openai.json')) # Load the JSON file
# items = data['items'] # Get the list of items

# for plugin in items: # Iterate over items, not data
#     url = plugin['manifest']['api']['url']
#     tool_name = plugin['namespace']
#     tools_mappings[tool_name] = url[:-len('/.well-known/openai.yaml')]

# print(tools_mappings)

valid_tools_info = []
all_tools_list = []

gr.close_all()

MAX_TURNS = 30
MAX_BOXES = MAX_TURNS * 2

return_msg = []
chat_history = ""

MAX_SLEEP_TIME = 40
valid_tools_info = {}

import gradio as gr
from tools_controller import load_valid_tools, tools_mappings

def load_tools():
    global valid_tools_info
    global all_tools_list
    try:
        valid_tools_info = load_valid_tools(tools_mappings)
        print(f"valid_tools_info: {valid_tools_info}")  # Debugging line
    except BaseException as e:
        print(repr(e))
    all_tools_list = sorted(list(valid_tools_info.keys()))
    print(f"all_tools_list: {all_tools_list}")  # Debugging line
    return gr.update(choices=all_tools_list)

def set_environ(OPENAI_API_KEY: str = "",
                WOLFRAMALPH_APP_ID: str = "",
                WEATHER_API_KEYS: str = "",
                BING_SUBSCRIPT_KEY: str = "",
                ALPHA_VANTAGE_KEY: str = "",
                BING_MAP_KEY: str = "",
                BAIDU_TRANSLATE_KEY: str = "",
                RAPIDAPI_KEY: str = "",
                SERPER_API_KEY: str = "",
                GPLACES_API_KEY: str = "",
                SCENEX_API_KEY: str = "",
                STEAMSHIP_API_KEY: str = "",
                HUGGINGFACE_API_KEY: str = "",
                AMADEUS_ID: str = "",
                AMADEUS_KEY: str = "",
                AWS_ACCESS_KEY_ID: str = "",
                AWS_SECRET_ACCESS_KEY: str = "",
                AWS_DEFAULT_REGION: str = "",
            ):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["WOLFRAMALPH_APP_ID"] = WOLFRAMALPH_APP_ID
    os.environ["WEATHER_API_KEYS"] = WEATHER_API_KEYS
    os.environ["BING_SUBSCRIPT_KEY"] = BING_SUBSCRIPT_KEY
    os.environ["ALPHA_VANTAGE_KEY"] = ALPHA_VANTAGE_KEY
    os.environ["BING_MAP_KEY"] = BING_MAP_KEY
    os.environ["BAIDU_TRANSLATE_KEY"] = BAIDU_TRANSLATE_KEY
    os.environ["RAPIDAPI_KEY"] = RAPIDAPI_KEY
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    os.environ["GPLACES_API_KEY"] = GPLACES_API_KEY
    os.environ["SCENEX_API_KEY"] = SCENEX_API_KEY
    os.environ["STEAMSHIP_API_KEY"] = STEAMSHIP_API_KEY
    os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
    os.environ["AMADEUS_ID"] = AMADEUS_ID
    os.environ["AMADEUS_KEY"] = AMADEUS_KEY
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
    
    if not tool_server_flag:
        start_tool_server()
        time.sleep(MAX_SLEEP_TIME)

    # Check if AWS keys are set and if so, configure AWS
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION:
        try:
            s3 = boto3.client('s3')
            s3.list_buckets()
            aws_status = "AWS setup successful"
        except NoCredentialsError:
            aws_status = "AWS setup failed: Invalid credentials"
    else:
        aws_status = "Keys set successfully"

    return gr.update(value="OK!"), aws_status

def show_avatar_imgs(tools_chosen):
    if len(tools_chosen) == 0:
        tools_chosen = list(valid_tools_info.keys())
    img_template = '<a href="{}" style="float: left"> <img style="margin:5px" src="{}.png" width="24" height="24" alt="avatar" /> {} </a>'
    imgs = [valid_tools_info[tool]['avatar'] for tool in tools_chosen if valid_tools_info[tool]['avatar'] != None]
    imgs = ' '.join([img_template.format(img, img, tool) for img, tool in zip(imgs, tools_chosen)])
    return [gr.update(value='<span class="">' + imgs + '</span>', visible=True), gr.update(visible=True)]

def answer_by_tools(question, tools_chosen, model_chosen):
    global return_msg
    return_msg += [(question, None), (None, '...')]
    yield [gr.update(visible=True, value=return_msg), gr.update(), gr.update()]
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')

    if len(tools_chosen) == 0:  # if there is no tools chosen, we use all todo (TODO: What if the pool is too large.)
        tools_chosen = list(valid_tools_info.keys())

    if len(tools_chosen) == 1:
        answerer = STQuestionAnswerer(OPENAI_API_KEY.strip(), OPENAI_BASE_URL, stream_output=True, llm=model_chosen)
        agent_executor = answerer.load_tools(tools_chosen[0], valid_tools_info[tools_chosen[0]],
                                             prompt_type="react-with-tool-description", return_intermediate_steps=True)
    else:
        answerer = MTQuestionAnswerer(OPENAI_API_KEY.strip(), OPENAI_BASE_URL,
                                      load_valid_tools({k: tools_mappings[k] for k in tools_chosen}),
                                      stream_output=True, llm=model_chosen)

        agent_executor = answerer.build_runner()

    global chat_history
    chat_history += "Question: " + question + "\n"
    question = chat_history
    for inter in agent_executor(question):
        if isinstance(inter, AgentFinish): continue
        result_str = []
        return_msg.pop()
        if isinstance(inter, dict):
            result_str.append("<font color=red>Answer:</font> {}".format(inter['output']))
            chat_history += "Answer:" + inter['output'] + "\n"
            result_str.append("...")
        else:
            try:
                not_observation = inter[0].log
            except:
                print(inter[0])
                not_observation = inter[0]
            if not not_observation.startswith('Thought:'):
                not_observation = "Thought: " + not_observation
            chat_history += not_observation
            not_observation = not_observation.replace('Thought:', '<font color=green>Thought: </font>')
            not_observation = not_observation.replace('Action:', '<font color=purple>Action: </font>')
            not_observation = not_observation.replace('Action Input:', '<font color=purple>Action Input: </font>')
            result_str.append("{}".format(not_observation))
            result_str.append("<font color=blue>Action output:</font>\n{}".format(inter[1]))
            chat_history += "\nAction output:" + inter[1] + "\n"
            result_str.append("...")
        return_msg += [(None, result) for result in result_str]
        yield [gr.update(visible=True, value=return_msg), gr.update(), gr.update()]
    return_msg.pop()
    if return_msg[-1][1].startswith("<font color=red>Answer:</font> "):
        return_msg[-1] = (return_msg[-1][0], return_msg[-1][1].replace("<font color=red>Answer:</font> ",
                                                                       "<font color=green>Final Answer:</font> "))
    yield [gr.update(visible=True, value=return_msg), gr.update(visible=True), gr.update(visible=False)]


def retrieve(tools_search):
    if tools_search == "":
        return gr.update(choices=all_tools_list)
    else:
        url = "http://127.0.0.1:8079/retrieve"
        param = {
            "query": tools_search
        }
        response = requests.post(url, json=param)
        result = response.json()
        retrieved_tools = result["tools"]
        return gr.update(choices=retrieved_tools)

def clear_retrieve():
    return [gr.update(value=""), gr.update(choices=all_tools_list)]


def clear_history():
    global return_msg
    global chat_history
    return_msg = []
    chat_history = ""
    yield gr.update(visible=True, value=return_msg)



def fetch_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return f"Tokenizer for {model_name} loaded successfully."
    except Exception as e:
        return f"Error loading tokenizer: {str(e)}"

# Add this function to handle the button click
import sky

# with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as demo:
with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=14):
                gr.Markdown("")

        with gr.Tab("Key setting"):
            OPENAI_API_KEY = gr.Textbox(label="OpenAI API KEY:", placeholder="sk-...", type="text")
            WOLFRAMALPH_APP_ID = gr.Textbox(label="Wolframalpha app id:", placeholder="Key to use wlframalpha", type="text")
            WEATHER_API_KEYS = gr.Textbox(label="Weather api key:", placeholder="Key to use weather api", type="text")
            BING_SUBSCRIPT_KEY = gr.Textbox(label="Bing subscript key:", placeholder="Key to use bing search", type="text")
            ALPHA_VANTAGE_KEY = gr.Textbox(label="Stock api key:", placeholder="Key to use stock api", type="text")
            BING_MAP_KEY = gr.Textbox(label="Bing map key:", placeholder="Key to use bing map", type="text")
            BAIDU_TRANSLATE_KEY = gr.Textbox(label="Baidu translation key:", placeholder="Key to use baidu translation", type="text")
            RAPIDAPI_KEY = gr.Textbox(label="Rapidapi key:", placeholder="Key to use zillow, airbnb and job search", type="text")
            SERPER_API_KEY = gr.Textbox(label="Serper key:", placeholder="Key to use google serper and google scholar", type="text")
            GPLACES_API_KEY = gr.Textbox(label="Google places key:", placeholder="Key to use google places", type="text")
            SCENEX_API_KEY = gr.Textbox(label="Scenex api key:", placeholder="Key to use sceneXplain", type="text")
            STEAMSHIP_API_KEY = gr.Textbox(label="Steamship api key:", placeholder="Key to use image generation", type="text")
            HUGGINGFACE_API_KEY = gr.Textbox(label="Huggingface api key:", placeholder="Key to use models in huggingface hub", type="text")
            AMADEUS_KEY = gr.Textbox(label="Amadeus key:", placeholder="Key to use Amadeus", type="text")
            AMADEUS_ID = gr.Textbox(label="Amadeus ID:", placeholder="Amadeus ID",
                                    type="text")
            AWS_ACCESS_KEY_ID = gr.Textbox(label="AWS Access Key ID:", placeholder="AWS Access Key ID", type="text")
            AWS_SECRET_ACCESS_KEY = gr.Textbox(label="AWS Secret Access Key:", placeholder="AWS Secret Access Key", type="text")
            AWS_DEFAULT_REGION = gr.Textbox(label="AWS Default Region:", placeholder="AWS Default Region", type="text")
            key_set_btn = gr.Button(value="Set keys!")


        with gr.Tab("Chat with Tool"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        with gr.Column(scale=0.85):
                            txt = gr.Textbox(show_label=False, placeholder="Question here. Use Shift+Enter to add new line.",
                                            lines=1)
                        with gr.Column(scale=0.15, min_width=0):
                            buttonChat = gr.Button("Chat")

                    chatbot = gr.Chatbot(show_label=False, visible=True)
                    buttonClear = gr.Button("Clear History")
                    buttonStop = gr.Button("Stop", visible=False)

                with gr.Column(scale=4):
                    with gr.Row():
                        tools_search = gr.Textbox(
                            lines=1,
                            label="Tools Search",
                            placeholder="Please input some text to search tools.",
                        )
                        buttonSearch = gr.Button("Reset search condition")
                    tools_chosen = gr.CheckboxGroup(
                        choices=all_tools_list,
                        # value=["chemical-prop"],
                        label="Tools provided",
                        info="Choose the tools to solve your question.",
                    )

            # TODO fix webgl galaxy backgroun
            # def serve_iframe():
            #     return "<iframe src='http://localhost:8000/shader.html' width='100%' height='400'></iframe>"

            # iface = gr.Interface(fn=serve_iframe, inputs=[], outputs=gr.outputs.HTML())

            key_set_btn.click(fn=set_environ, inputs=[
            OPENAI_API_KEY,
            WOLFRAMALPH_APP_ID,
            WEATHER_API_KEYS,
            BING_SUBSCRIPT_KEY,
            ALPHA_VANTAGE_KEY,
            BING_MAP_KEY,
            BAIDU_TRANSLATE_KEY,
            RAPIDAPI_KEY,
            SERPER_API_KEY,
            GPLACES_API_KEY,
            SCENEX_API_KEY,
            STEAMSHIP_API_KEY,
            HUGGINGFACE_API_KEY,
            AMADEUS_ID,
            AMADEUS_KEY,
        ], outputs=key_set_btn)
        key_set_btn.click(fn=load_tools, outputs=tools_chosen)

        tools_search.change(retrieve, tools_search, tools_chosen)
        buttonSearch.click(clear_retrieve, [], [tools_search, tools_chosen])

        txt.submit(lambda: [gr.update(value=''), gr.update(visible=False), gr.update(visible=True)], [],
                [txt, buttonClear, buttonStop])
        inference_event = txt.submit(answer_by_tools, [txt, tools_chosen], [chatbot, buttonClear, buttonStop])
        buttonChat.click(answer_by_tools, [txt, tools_chosen], [chatbot, buttonClear, buttonStop])
        buttonStop.click(lambda: [gr.update(visible=True), gr.update(visible=False)], [], [buttonClear, buttonStop],
                        cancels=[inference_event])
        buttonClear.click(clear_history, [], chatbot)

# demo.queue().launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=7001)
demo.queue().launch(share=True)


