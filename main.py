from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import os
import openai
import tiktoken

with open("multinomial_naive_bayes_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("cv.pkl", "rb") as f:
    cv = joblib.load(f)

key=""
openai.api_key=key
system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit = 4096
conversation = [system_message]


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def chatGPT(prompt):
    global conversation
    conversation.append({"role": "user", "content": prompt})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while conv_history_tokens + max_response_tokens >= token_limit:
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=1,
        max_tokens=max_response_tokens,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    return str(response['choices'][0]['message']['content'])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')


#
@app.route('/news', methods=['GET', 'POST'])
def news():
	try:
		pr = request.args.get('prompt')
		req=[pr]
		data = cv.transform(req).toarray()
		res = model.predict(data)[0] #['vrais']
		return res
	except Exception:
		return "Error"


@app.route('/imggen', methods=['GET', 'POST'])
def imggen():
	try:
		pr = request.args.get('prompt')
		#print(pr) #the earthe after years of bad climat
		result = openai.Image.create(
	        prompt=pr,
	        n=1,
	        size="512x512"
	    )
	    #['data'][0]['url']
		return result['data'][0]['url'] #jsonify(result)
	except Exception:
		return "static/img/im1.png" #render_template('index.html')


@app.route('/assistant', methods=['GET', 'POST'])
def assistant():
	try:
		pr = request.args.get('prompt')
		result = chatGPT(pr)
		return result
	except Exception:
		return "Error"



app.run(host='0.0.0.0', port=5000)

