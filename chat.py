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
import random
import csv
from fastapi.responses import FileResponse
import requests
# Step 1: Import the logging module
import logging
# Step 2: Set up a logger instance
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('logs.log')
# Step 3: Configure logging level and output format
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def grammar_check(text):
    # إعداد النص للتحقق من الأخطاء القواعدية
    payload = {'language': 'en-US', 'text': text}

    # إرسال طلب التحقق إلى خدمة LanguageTool
    response = requests.post('https://api.languagetool.org/v2/check', data=payload)

    # تحليل الاستجابة
    errors = []
    if response.status_code == 200:
        data = response.json()
        for mistake in data['matches']:
            if mistake['message']not in ['This sentence does not start with an uppercase letter.','Possible typo: you repeated a whitespace']:
                errors.append(mistake['message'])

    return errors
def ZbotChecker(text):
    errors=grammar_check(text)
    if errors:
        prompt="Correct “Text:{}” to standard English and place the results in “Correct Text:”".format(text)
        return Zbot(prompt,"text-davinci-003",1)
    else:
        return False


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
def Zbot(prompt,COMPLETIONS_MODEL,temperature):
        bot_response = openai.Completion.create(
            prompt=prompt,
            temperature=1,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=COMPLETIONS_MODEL
        )["choices"][0]["text"].strip(" \n")
        return bot_response


def conversation(user_response):
    user_response,user_name=user_response.split('-#-')
    data = load_dict_from_json('data.json')
    user=data[user_name]
    def convert_to_short_parts(response, max_length):
        parts = []
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(?<!\d\.)\s"
        sentences = re.split(pattern, response)
        current_part = ""
        for sentence in sentences:
            if len(current_part) + len(sentence) <= max_length:
                current_part += sentence
            elif sentence.endswith('.'):
                current_part += sentence
                parts.append(current_part)
                current_part = ""
            else:
                parts.append(current_part)
                current_part = sentence
        if current_part != '':
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
        user['correct']=False
        result = llm_chain.predict(question=msg)
        with get_openai_callback() as cb:
            result = llm_chain.predict(question=msg)
            user['total_cost']+=cb.total_cost
            user['total_tokens']+=cb.total_tokens
        
        result = result.replace('Zbot:', '', -1).replace('AI:', '', -1).replace('Zbot:', '', -1).replace('<', '[<nocode]', -1)
        end_time = time.time()  # End the timer
        user['total_chat_duration'] = (end_time - user['start_time'])/60
        temp2=user['t1']
        user['t1'] = user['t1'] + '\nuser:' + msg + '\nZbot:' + result
        try :
            prompt_template = PromptTemplate(input_variables=["chat_history", "question"], template=user['template'] + user['t1'] + user['t2'])
        except:
            user['t1']=temp2
            user['t1'] = user['t1'] + '\nuser:' + msg + '\nZbot:' + 'here is your code ^__^'
        data[user_name]=user
        save_dict_to_json(data, 'data.json')
        correct=ZbotChecker(msg)
        if correct:
            user['correct']=correct.replace('Correct','Corrected')
        return result


    

    def check(bot_response, user_response, problem):
        prompt = """check if "{}" in following conversation ? return 'yes' if it is true else return 'no' " .\n Bot: {} \nUser: {}""".format(
            problem, bot_response.strip(), user_response.strip())
        temp =Zbot(prompt,"text-davinci-003",1)
        if "no".lower() in temp.lower():
            prompt = """give user example  response for this 'Bot:{}'  """.format(bot_response)
            result = Zbot(prompt,"text-davinci-003",1)
            return result
        else:
            return False


    
    
    
    if user['step'] == 'step1':
        bot_response = check('What is your name?', user_response,
                             'user says his name no matter if he write his name in small letters')
        if bot_response:
            return ['This is an example for good response:\n' + bot_response+'✏️📝🔍📚📖']
        else:
            user['history'].append(user_response)
            user['full_name'] = user_response
            user['step'] = 'step2'
            bot_response = ["let's start by sharing with me your interest, so we can have a better journey together.","What are your interests?👋"]
                
            user['history'].append(bot_response)
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return bot_response
    
    if user['step'] == 'step2':
        bot_response = check('what are your intersts?', user_response, 'User write his interests')
        if bot_response:
            return ['This is an example for good response:\n' + bot_response+'✏️📝🔍📚📖']
        else:
            user['history'].append(user_response)
            user['interest'] = user_response
            user['step'] = 'step3'
            user['history'].append(bot_response)
            user['template'] = user['template'].format(user['full_name'],user['interest'])
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            temp = warmup('hey!')
            edit_result = convert_to_short_parts(temp, 30)
            edit_result = edit_sentences(edit_result)
            data[user_name]=user
            save_dict_to_json(data, 'data.json')
            return edit_result

    if user['step'] == 'step3' and user_response.strip() != 'RESET' and user_response.strip() != 'START_STUDY_PLAN':
        temp = warmup(user_response)
        edit_result = convert_to_short_parts(temp, 30)
        edit_result = edit_sentences(edit_result)
        if user['correct']:
            a='<span style="color: green;">'+user['correct']+'✏️📝🔍📚📖'+'</span>'
            edit_result.insert(0,a)
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
def home(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /")
        user = {
        "full_name": None,
        "interest": None,
        "total_chat_duration": 0,
        "step": "step1",
        "history": [],
        "total_cost":0.0,
        "total_tokens":0.0,
        "start_time":0.0,
        "correct":False,
        "t1": """
            \n
            history:
                user:please act as my friend to chat about any topic.Use many Emojis for each response(5 at least).
                Zbot:Sure.
                user:Imagine yourself as a friend with his views and interests.
                Zbot:ok.
                user:if I do not have a topic or ideas,suggest anything related to my interests.
                Zbot:Sure.
                user:please just response to me without more talking or repeating.Don't repeat a question you've asked before.
                Zbot:Sure,I will.
                user:Use short response always.do not repeat any thing from history of chat.your response should be less than 15 words.
                Zbot:Sure,I will.
                user:if I ask you "who are you?" tell me about you. "Hello my friend, my name Zbot 😊 and I'm here to chat with you.⚙️🤖💬"
                Zbot:ok , I tell you about Zbot.
                user:Firstly respond to me and ask me "how are you doing?"
                Zbot:ok. I well.
                user:do not suggest online resources.
                Zbot:Sure.
                user:if I was in bad mood or not ready to chat tell me joke or advice related to my interest.stop chatting until I will be ok.
                Zbot:ok .I well.
                user:Respond by relying on history of conversation.
                Zbot:ok.
                user:do not return any code response,return "I am sorry, I can not write a code 😔😞".
                Zbot:sure,code is not available 😔😞.
                user:can you write a code".
                Zbot:No 😔😞.
                user:chat me using my name.
                Zbot:Sure.
        """,
        "t2": """
            {chat_history}
            user: {question}
            Zbot:
        """,
        "template": """
            as a Freind called "Zbot" who has same interests and goals.respond to user in smart way. 
            user name is {},user interests  are  {}.
        """
    }
        

        username =random.randint(1,9999999)
        data = load_dict_from_json('data.json')
        user['start_time']=time.time()
        data[username]=user
        save_dict_to_json(data, 'data.json')
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
@app.get("/getChatBotResponse")
def get_bot_response(msg: str,request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /getChatBotResponse")
        result = conversation(msg)
        return result
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
    
@app.get("/report_for_zu")
def send(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /report_for_zu")  
        data = load_dict_from_json('data.json')
        return templates.TemplateResponse("redirect.html", {"request": request,"users":data.values()})
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
    

@app.get("/save_report_for_zu")
def send(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /save_report_for_zu")
        data = load_dict_from_json('data.json')
        users = data.values()

        # Specify the file path for the CSV
        csv_file_path = 'report.csv'

        # Define the fieldnames for the CSV
        fieldnames = ['full_name', 'total_cost', 'total_tokens', 'total_chat_duration']

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            # Write each user's data to the CSV file
            for user in users:
                if user['full_name']:
                    writer.writerow({
                        'full_name': user['full_name'],
                        'total_cost': user['total_cost'],
                        'total_tokens': user['total_tokens'],
                        'total_chat_duration': user['total_chat_duration']
                    })

        # Return the CSV file as a response
        return FileResponse(csv_file_path, filename='report.csv')
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
@app.get("/logs")
def send(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /logs")
        return FileResponse('logs.log', filename='logs.log')
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response 
@app.get("/reset_tarek")
def reset(request: Request):
    try:
        # Logging an informational message
        logger.info("Received a GET request to /reset_tarek")
        save_dict_to_json({}, 'data.json')      
    except Exception as e:
        # Logging an error message
        logger.error(f"An error occurred: {str(e)}")
        # Handle the exception and return an appropriate response
      
if __name__ == "__main__":
    uvicorn.run("chat:app",reload=True)
