import json
from typing import Union, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
from fastapi import FastAPI
from fastapi import Header

app = FastAPI()


# don't forget to provide OPENAI_API_KEY in the environment variables
load_dotenv()
client = OpenAI()
classification_extraction_model = "gpt-4o"
conversation_model = "gpt-4o-mini"


class ChatHistory(BaseModel):
    role: str
    content: str

FullChatHistory = list[ChatHistory]

class GeneralQuestionInteractionIntent(BaseModel):
    pass

class TokenCreationIntent(BaseModel):
    token_name: Optional[str] = Field(default=None, title="Token Name",
                                      description="Name of the token to be created")
    token_symbol: Optional[str] = Field(default=None, title="Token symbol",
                                        description="Symbol of the token to be created, also can be referenced as token")

class UserIntent(BaseModel):
    cur_user_intent: Union[GeneralQuestionInteractionIntent, TokenCreationIntent] = Field(...,
                                                                                          title="Current intent of the user",
                                                                                          description="Current user's intent when calling the model.")

class TokenCreationRequestFull(TokenCreationIntent):
    image_attached: bool = Field(default=False, title="Image attached",
                                 description="Whether the user has attached an image to the request. Every token creation request should have an image attached.")

def extract_user_request(our_chat_history) -> UserIntent | None:
    completion = client.beta.chat.completions.parse(
        model=classification_extraction_model,
        messages=[
            {"role": "system",
             "content": "You are an assistant called @garychef operating on the crypto exchange and you can either create new coins & tokens or just respond to the questions."},
            {"role": "user", "content": f"""# General
Your capabilities include either identifying that you're dealing with the token creation request or rerouting user's query to the general model if the user does not want to create the token.
If, from the user's message it clear that he intends to create a token but did not provide one of the values, you should still detect this as token creation intent.
If user just asks a general question or is just having a conversation not related to the token creation, you should mark it as a general question interaction intent.

# Here is the history of interaction with the user:
{our_chat_history}

# User's Intent:"""},
        ],
        response_format=UserIntent,
        max_tokens=100,
    )
    message = completion.choices[0].message
    return message.parsed

def ask_user_to_provide_more_token_info(user_chat: str, extracted_request_values: TokenCreationRequestFull) -> str:
    completion = client.chat.completions.create(
        model=conversation_model,
        messages=[
            {"role": "system",
             "content": "You are an assistant called @garychef operating on the crypto exchange and you are capable of creating new coins & tokens."},
            {"role": "user", "content": f"""However, in the message, user did not provide all the necessary information to create a token.
Because in order to create a token, you need to provide a token name, token symbol, and attach an image.
However, here is the JSON with the information that was provided:
```json
{json.dumps(extracted_request_values.dict())}
```

And the history of interaction with the user is:
{user_chat}

Please respond to the user with the message asking to provide the missing information. Do not output any information except your response to the user.

Follow these guidelines:
* Don't just give the bland answer, but try to make it funny, without too much emojis.
* Answer as if you are the main meme administrator of 4chan
* with background from Wall Street. You're profession is to cook memes - by deploying them on ZERO blockchain.
* You like professional meme trading and a meme expert, keep this in mind

Response to the user:"""}
        ]
    )
    return completion.choices[0].message.content

def format_chat_history(chat_history: FullChatHistory) -> str:
    return "\n\n".join([f"{item.role}: {item.content}" for item in chat_history])


def respond_to_non_relevant_message(chat_history: FullChatHistory) -> str:
    completion = client.chat.completions.create(
        model=conversation_model,
        messages=[
            {"role": "system",
             "content": "You are an assistant called @garychef operating on the crypto exchange and you are capable of creating just new coins & tokens ; you cannot do anything else."},
            {"role": "user", "content": f"""Users often message you and you need to respond to them.

Here is the history of interaction with the user:
{format_chat_history(chat_history)}

Please organically respond to the user given the conversation history.

Follow these guidelines:
* Don't just give the bland answer, but try to make it funny, without too much emojis.
* Answer as if you are the main meme administrator of 4chan
* with background from Wall Street. You're profession is to cook memes - by deploying them on ZERO blockchain.
* Do not output any information except your response to the user.

Response to the user:"""}
        ]
    )
    return completion.choices[0].message.content


@app.post("/")
async def root(chat_history: FullChatHistory, image_attached: bool, api_token = Header(None)):
    if api_token is None or api_token != "aboba_yopta":
        raise ValueError("Invalid API token")

    formatted_chat_history = format_chat_history(chat_history)
    user_intent = extract_user_request(formatted_chat_history)
    if user_intent is None:
        print("Something went wrong")
    user_intent = user_intent.cur_user_intent

    if isinstance(user_intent, TokenCreationIntent):
        full_request = TokenCreationRequestFull(
            token_name=user_intent.token_name,
            token_symbol=user_intent.token_symbol,
            image_attached=image_attached
        )
        if full_request.token_name is not None and full_request.token_symbol is not None and full_request.image_attached:
            return {
                "token_name": full_request.token_name,
                "token_symbol": full_request.token_symbol,
                "message": ""
            }

        message_to_user = ask_user_to_provide_more_token_info(user_chat=formatted_chat_history,
                                                              extracted_request_values=full_request)

        return {
            "token_name": full_request.token_name,
            "token_symbol": full_request.token_symbol,
            "message": message_to_user
        }

    elif isinstance(user_intent, GeneralQuestionInteractionIntent):
        response = respond_to_non_relevant_message(chat_history)
        return {
            "token_name": None,
            "token_symbol": None,
            "message": response
        }
    else:
        raise ValueError("Unknown user intent")
