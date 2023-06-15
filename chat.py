from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import json 
import numpy as np
from fastapi import FastAPI, Request, Form
import random
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
import os 
from fastapi.staticfiles import StaticFiles
from langchain.callbacks import get_openai_callback
import re
import time
import sys
import json

# Function to load a dictionary from a JSON file
def load_dict_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to save a dictionary to a JSON file
def save_dict_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
#open_ai_model
temp='%s%k%-%N%V%b%i%n%T%V%Y%L%a%W%N%T%M%9%I%o%u%x%z%T%3%B%l%b%k%F%J%y%h%0%n%P%X%A%s%J%h%7%8%t%W%h%a%2%f%d%z'
api_key=""
for i in range(1,len(temp),2):
    api_key+=temp[i]
os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key
COMPLETIONS_MODEL = "text-davinci-002"


def conversation(user_response):
    user_response,user_name=user_response.split('-#-')
    data = load_dict_from_json('data.json')
    user=data[user_name]
    def convert_to_short_parts(response, max_length):
        parts = []
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        sentences = re.split(pattern, response)
        current_part = ""
        for sentence in sentences:
            if len(current_part) + len(sentence) <= max_length:
                current_part += sentence
            else:
                parts.append(current_part)
                current_part = sentence
        if current_part!='':
            parts.append(current_part)
        parts = list(filter(lambda item: item != '', parts))
        return parts

    def edit_sentences(sentences):
            def is_emoji(character):
                ascii_value = ord(character)
                return 1000 <= ascii_value  # نطاق السمايلات في ترميز ASCII

            result = []
            previous_sentence = ""

            for s in range(len(sentences)):
                temp=""
                for i in range(len(sentences[s])):
                    if is_emoji(sentences[s][i]):
                        temp+=sentences[s][i]
                    else:
                        break
                if temp!="":
                    sentences[s-1]=sentences[s-1]+temp
                    sentences[s]=sentences[s][len(temp):]
            sentences = list(filter(lambda item: item != '', sentences))         
            return sentences
    def warmup(msg):
        prompt_template = PromptTemplate(input_variables=["chat_history", "question"], template=user['template'] + user['t1'] + user['t2'])
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm_chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100, n=1),
            prompt=prompt_template,
            verbose=False,
            memory=memory
        )
        result = llm_chain.predict(question=msg)
        #start_time = time.time()  # Start the timer
        #with get_openai_callback() as cb:
        #    result = llm_chain.predict(question=msg)
        #    user['bills'].append(cb)
        #end_time = time.time()  # End the timer

        result = result.replace('A2ZBot:', '', -1).replace('AI:', '', -1).replace('A2Zbot:', '', -1)
        #chat_time = end_time - start_time
        #user['total_chat_duration'] += chat_time
        #last_response_time = end_time
        user['t1'] = user['t1'] + '\nuser:' + msg + '\nA2Zbot:' + result
        data[user_name]=user
        save_dict_to_json(data, 'data.json')
        return result


    def A2ZBot(prompt):
        bot_response = openai.Completion.create(
            prompt=prompt,
            temperature=0.9,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=COMPLETIONS_MODEL
        )["choices"][0]["text"].strip(" \n")
        return bot_response


    def check(bot_response, user_response, problem):
        prompt = """check if "{}" in following conversation ? return 'yes' if it is true else return 'no' " .\n Bot: {} \nUser: {}""".format(
            problem, bot_response.strip(), user_response.strip())
        temp = A2ZBot(prompt)
        if "no".lower() in temp.lower():
            prompt = """give user example  response for this 'Bot:{}'  """.format(bot_response)
            result = A2ZBot(prompt)
            return result
        else:
            return False


    
    
    if user['step'] == 'step1':
        user['step'] = 'step2'
        bot_response = "What is your name?"
        user['history'].append(bot_response)
        data[user_name]=user
        save_dict_to_json(data, 'data.json')
        return [bot_response]

    if user['step'] == 'step2':
        bot_response = check(user['history'][-1], user_response,
                             'user says his name no matter if he write his name in small letters')
        if bot_response == "xxxxx":
            return ['This is an example for good response:\n' + bot_response]
        else:
            user['history'].append(user_response)
            user['full_name'] = user_response
            user['step'] = 'step3'
            bot_response = """What is your current english level:
                    <ul>
                        <li>A1</li>
                        <li>A2</li>
                        <li>B1</li>
                        <li>B2</li>
                        <li>C1</li>
                        <li>C2</li>
                    </ul>
                
                """
            user['history'].append(bot_response)
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return [bot_response]

    if user['step'] == 'step3':
        bot_response = check(user['history'][-1], user_response, 'User must to write his English Level from Bot options ')
        if bot_response == "xxxxx":
            return ['This is an example for good response:\n' + bot_response]
        else:
            user['history'].append(user_response)
            user['level'] = user_response
            user['step'] = 'step4'
            bot_response = """
                Please choose one or two paths from the following pathes: 
                    
                    <ul >
                        <li>Travel</li>
                <li>Business</li>
                <li>Fun/communication</li>
                <li>Education</li>
                <li>Default,General English</li> 
                    </ul>
                """

            user['history'].append(bot_response)
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return [bot_response]

    if user['step'] == 'step4':
        bot_response = check(user['history'][-1], user_response, 'User write his English Path from Bot options')
        if bot_response == "xxxxx":
            return ['This is an example for good response:\n' + bot_response]
        else:
            user['history'].append(user_response)
            user['path'] = user_response
            user['step'] = 'step5'
            bot_response = """
                what are your interests?
                    
                    <ul>
                        <li>Sport </li> 
                    <li>Art </li> 
                    <li> History </li> 
                    <li> Technology </li> 
                    <li> Gaming </li> 
                    <li> Movies </li> 
                    <li> Culture </li> 
                    <li> Management </li> 
                    <li>Science </li> 
                    <li>  Adventure </li> 
                    <li> Space </li> 
                    <li>Cooking </li> 
                    <li> Reading </li> 
                    <li> Lifestyle </li>
                    <li> ... </li> 
                    </ul>
                """
            user['history'].append(bot_response)
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return [bot_response]

    if user['step'] == 'step5':
        bot_response = check(user['history'][-1], user_response, 'User write his interests')
        if bot_response == "xxxxx":
            return ['This is an example for good response:\n' + bot_response]
        else:
            user['history'].append(user_response)
            user['interest'] = user_response
            user['step'] = 'step6'
            bot_response = """welcom to A2Zbot,let's start.
                    """
            user['history'].append(bot_response)
            user['template'] = user['template'].format(user['full_name'], user['level'], user['path'] + ' ' + user['interest'])
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return [bot_response]

    if user['step'] == 'step6' and user_response.strip() != 'RESET' and user_response.strip() != 'START_STUDY_PLAN':
        temp = warmup(user_response)
        edit_result = convert_to_short_parts(temp, 30)
        edit_result = edit_sentences(edit_result)
        data[user_name]=user
        save_dict_to_json(data, 'data.json')
        return edit_result
    

    
#openai_model

script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")

app = FastAPI()


templates = Jinja2Templates(directory="")
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")



@app.get("/", response_class=HTMLResponse)
def login(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def process_login(request: Request):
    user = {
    "current_user": None,
    "user_data": None,
    "email": None,
    "full_name": None,
    "level": None,
    "path": None,
    "interest": None,
    "bills": [],
    "total_chat_duration": 0,
    "step": "step1",
    "history": [],
    "vocabs": [],
    "messages": [],
    "last_response_time": None,
    "t1": """
        \n
        history:
            user:please act as my friend to chat about any topic.Use many Emojis for each response(5 at least).chat me using my name.
            A2Zbot:Sure.
            user:if I do not have a topic or ideas,suggest anything related to my interests.
            A2Zbot:Sure.
            user:Don't repeat a question you've asked me before like "How are you doing".
            A2Zbot:Sure.
            user:do not finish evrey response by question.act as a good listener.
            A2Zbot:Sure.
            user:please just response to me without more talking or repeating.Don't repeat a question you've asked before.
            A2Zbot:Sure,I will.
            user:Use short response always.do not repeat any thing from history of chat.your response should be less than 15 words.
            A2Zbot:Sure,I will.
            user:if I ask you "who are you?" tell me about you. "You are my friend called A2Zbot ,your goal is helping me to learn english"
            A2Zbot:ok , I tell you about A2Zbot.
            user:Firstly respond to me and ask me "how are you doing?"
            A2Zbot:ok. I well.
            user:if I suggest another topic do not change it please.and discuse me about current topic.do not suggest online resources.
            A2Zbot:Sure.
            user:if I suggest another topic do not change it please.
            A2Zbot:Sure.
            user:if I was in bad mood or not ready to chat tell me joke or advice related to my interest.stop chatting until I will be ok.
            A2Zbot:ok .I well.
            user:can you tell me about grammar and spelling mistakes if I had.
            A2Zbot:sure ,I will check evrey single response and correct your mistake then continue to chatting.
            user:Respond by relying on history of conversation.
            A2zbot:ok.
    """,
    "t2": """
        {chat_history}
        user: {question}
        A2Zbot:
    """,
    "template": """
        as a Freind called "A2Zbot" who has same interests and goals.respond to user in smart way. 
        user name is {},english level is {},interests and goals are  {}.
    """
}


    form_data = await request.form()
    username = form_data["username"]
    data = load_dict_from_json('data.json')
    data[username]=user
    save_dict_to_json(data, 'data.json')
    unique_link = f"/home/{username}"
    return templates.TemplateResponse("redirect.html", {"request": request, "link": unique_link})


@app.get("/home/{username}", response_class=HTMLResponse)
def home(request: Request, username: str):
    return templates.TemplateResponse("home.html", {"request": request, "username": username})
@app.get("/getChatBotResponse")
def get_bot_response(msg: str,request: Request):
    try: 
        result = conversation(msg)
        return result
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        try:
            error_details = f"Exception Type: {exc_type}\nException Value: {exc_value}\nTraceback: {exc_traceback}"
            return [error_details,str(sessions.keys())] 
        except:
            return ["empty data"]

if __name__ == "__main__":
    uvicorn.run("chat:app", reload=True)
