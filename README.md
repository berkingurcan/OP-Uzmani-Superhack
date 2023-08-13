# OP Uzmani AI Chatbot

The **OP Uzmani AI Chatbot** is a specialized tool tailored to support developers working within the OP-Stack environment. Powered by the cutting-edge GPT language model developed by OpenAI, it has undergone extensive training encompassing a wide spectrum of knowledge and expertise related to OP-Stack.

Unlike conventional chatbots, this unique AI-powered chatbot is equipped with the most up-to-date information pertaining to OP Stack and offers an unparalleled depth of understanding in the realm of OP-Stack.

## Project Overview

The **OP-Stack Uzmani** project endeavors to empower developers within the OP Stack ecosystem by introducing an advanced chatbot driven by artificial intelligence. This pioneering chatbot is designed to deliver information about OP Stack, instant code insights and debugging assistance, facilitating smoother and more efficient development experiences for OP Stack developers. By harnessing the capabilities of AI technology, the chatbot aims to enhance productivity and equip developers with the tools to effectively overcome challenges.

## Key Features

The OP-Stack AI Chatbot boasts a range of essential features:

- **Integration with OP Stack Documentation:** Seamlessly integrating with existing OP Stack tools and documentation, the chatbot becomes an invaluable resource for developers. It provides effortless access to pertinent information, helping developers navigate the intricate OP Stack ecosystem and fully capitalize on its potential.

## Future Improvements

- **Embed github repositories and example codes**
- **Create second index layer for secondary sources**
- **Also upload Optimisim documentation**
- **Create memory buffer for chatbot**
- **UI Improvements**

## Inclusive Collaboration

The **OP-Stack AI Chatbot** project warmly welcomes contributions from developers of all proficiency levels. Whether you're an aspiring protocol developer, a dApp enthusiast, or someone in between, your input is invaluable. By rallying the Optimism community together, we can harness the advanced AI capabilities of the chatbot to encourage collaboration and expedite development within the ecosystem.

## How To Run Project

Clone this repository

#### Back End

- You should have Python, virtualenv and Django installed.
  
After cloning repository go to backend folder
```cd backend```
Create a virtual env:
```
virtualenv venv
```

Activate venv:
```
source venv/bin/activate
```

Install requirements:
```pip install -r requirements.txt```

In chatbot folder, create .env file and fill these variables:
```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
```

You can get PINECONE API KEY in Pinecone App and you can get Open AI API KEY from your Open AI account.

Run the server:
```python manage.py runserver```

Now you can use ```http://127.0.0.1:8000/chatbot/send-query/```API for chatting :D

#### Client side

Create ```.env.local```file and copy and paste it(or your deployed backend API route):
```NEXT_PUBLIC_GETAPI='http://127.0.0.1:8000/chatbot/send-query/'```

Run:
```yarn```
Then,
```pnpm run dev```




