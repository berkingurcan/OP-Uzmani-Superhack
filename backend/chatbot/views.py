import os
import openai
import tiktoken
import json
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from langchain.document_loaders import UnstructuredMarkdownLoader


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

def get_log(request):
    data = {
        'message': 'working'
    }
    return JsonResponse(data)

@csrf_exempt
def get_completion( request, 
                    model="gpt-3.5-turbo",
                    temperature=0, 
                    max_tokens=500):
    if request.method == 'POST':
        # Parse the JSON data from the request body
        data = json.loads(request.body)
        prompt = data.get("messages")
    else:
        return HttpResponse("Invalid request method")
    
    
    messages = prompt
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
        max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
    )
    return HttpResponse(response.choices[0].message["content"])