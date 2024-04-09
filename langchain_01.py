####################################################################
# 랭체인(langchain)의 OpenAI GPT 모델(ChatOpenAI) 사용법 (1)
# https://teddylee777.github.io/langchain/langchain-tutorial-01/
####################################################################

# openai 파이썬 패키지 설치
# pip install openai langchain

# 2024-04-07일 기준.
# % pip list | grep openai
# langchain-openai           0.0.8
# openai                     1.14.3 

import os, sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI()

# 사용 가능한 모델 리스트 출력
#model_list = sorted([m['id'] for m in client.models.list()['data']])

#### model_list = sorted([m.id for m in client.models.list().data])
#### for m in model_list:
####     print(m)

''' 
babbage-002
dall-e-2
dall-e-3
davinci-002
gpt-3.5-turbo
gpt-3.5-turbo-0125
gpt-3.5-turbo-0301
gpt-3.5-turbo-0613
gpt-3.5-turbo-1106
gpt-3.5-turbo-16k
gpt-3.5-turbo-16k-0613
gpt-3.5-turbo-instruct
gpt-3.5-turbo-instruct-0914
gpt-4
gpt-4-0125-preview
gpt-4-0613
gpt-4-1106-preview
gpt-4-1106-vision-preview
gpt-4-turbo-preview
gpt-4-vision-preview
text-embedding-3-large
text-embedding-3-small
text-embedding-ada-002
tts-1
tts-1-1106
tts-1-hd
tts-1-hd-1106
whisper-1
'''


####################################################################
# ChatOpenAI
# OpenAI 사의 채팅 전용 Large Language Model(llm)
# temperature : 사용할 샘플링 온도는 0과 2 사이에서 선택합니다. 0.8과 같은 높은 값은 출력을 더 무작위하게 만들고, 0.2와 같은 낮은 값은 출력을 더 집중되고 결정론적으로 만듭니다.
# max_tokens : 채팅 완성에서 생성할 토큰의 최대 개수입니다.
####################################################################

# 원본코드 
#from langchain.chat_models import ChatOpenAI

#langchain==0.2.0부터 langchain에서 가져오기는 더 이상 지원되지 않습니다. 대신 langchain-community에서 가져옵니다.
# from langchain_community.chat_models import ChatOpenAI

#`langchain_community.chat_models.openai.ChatOpenAI` 클래스는 langchain-community 0.0.10에서 더 이상 사용되지 않으며 0.2.0에서 제거될 예정입니다. 
# 클래스의 업데이트된 버전이 langchain-openai 패키지에 있으므로 대신 사용해야 합니다. 
# 이를 사용하려면 `pip install -U langchain-openai`를 실행하고 `from langchain_openai import ChatOpenAI`로 가져옵니다.

'''
% pip list | grep langchain
langchain                  0.1.12
langchain-community        0.0.28
langchain-core             0.1.32
langchain-experimental     0.0.54
langchain-openai           0.0.8
langchain-text-splitters   0.0.1
'''

from langchain_openai import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(temperature=0,               # 창의성 (0.0 ~ 2.0) 
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-3.5-turbo',  # 모델명
                )

# 질의내용
question = '대한민국의 수도는 뭐야?'

# 질의
# the function `predict` was deprecated in LangChain 0.1.7 and will be removed in 0.2.0. Use invoke instead.
#print(f'[답변]: {llm.predict(question)}')

# https://python.langchain.com/docs/changelog/core/
# BaseChatModel methods __call__, call_as_llm, predict, predict_messages. Will be removed in 0.2.0. Use BaseChatModel.invoke instead.
# BaseChatModel methods apredict, apredict_messages. Will be removed in 0.2.0. Use BaseChatModel.ainvoke instead.
# BaseLLM methods __call__, predict, predict_messages. Will be removed in 0.2.0. Use BaseLLM.invoke` instead.
# BaseLLM methods apredict, apredict_messages. Will be removed in 0.2.0. Use BaseLLM.ainvoke instead.

#### print(f'[답변]: {llm.invoke(question)}')

####################################################################
# 프롬프트 템플릿의 활용
# PromptTemplate : 사용자의 입력 변수를 사용하여 완전한 프롬프트 문자열을 만드는 데 사용되는 템플릿입니다
# 사용법
# - template : 템플릿 문자열입니다. 이 문자열 내에서 중괄호 {}는 변수를 나타냅니다.
# - input_variables : 중괄호 안에 들어갈 변수의 이름을 리스트로 정의합니다.
# input_variables : input_variables는 PromptTemplate에서 사용되는 변수의 이름을 정의하는 리스트입니다.
# 사용법
# - 리스트 형식으로 변수 이름을 정의합니다.
####################################################################

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 질문 템플릿 형식 정의
template = '{country}의 수도는 뭐야?'

# 템플릿 완성
prompt = PromptTemplate(template=template, input_variables=['country'])

####################################################################
# LLMChain 객체
# LLMChain : LLMChain은 특정 PromptTemplate와 연결된 체인 객체를 생성합니다
# 사용법
# - prompt: 앞서 정의한 PromptTemplate 객체를 사용합니다.
# - llm: 언어 모델을 나타내며, 이 예시에서는 이미 어딘가에서 정의된 것으로 보입니다.
####################################################################

# 연결된 체인(Chain)객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

####################################################################
# 1. run() : 함수로 템플릿 프롬프트 실행
# just use invoke(), as __call__() and run() is deprecated in LangChain 0.1.0 and will be removed in 0.2.0
####################################################################
# 체인 실행: run() 
#print(llm_chain.run(country='일본'))
#print(llm_chain.invoke(country='일본'))

# The function `__call__` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
#print("1 -> {}".format(llm_chain("일본")))

# LangChainDeprecationWarning: The function `run` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
#print("2 -> {}".format(llm_chain.run("일본")))

# 체인 실행: invoke() 
#### print("3 -> {}".format(llm_chain.invoke("일본")))

'''
답변 : 
{'country': '일본', 'text': '일본의 수도는 도쿄입니다.'}
'''

# 체인 실행: invoke() : 파라미터값을 dictionary로 전달..
#print(llm_chain.run(country='캐나다'))
#### print(llm_chain.invoke({'country': '캐나다'}))

'''
{'country': '캐나다', 'text': '캐나다의 수도는 오타와(Ottawa)입니다.'}
'''


####################################################################
# 2. apply() : 함수로 여러개의 입력을 한 번에 실행
# text 키 값으로 결과 뭉치가 반환되었음을 확인할 수 있습니다.
# 이를 반복문으로 출력
####################################################################
input_list = [
    {'country': '호주'},
    {'country': '중국'},
    {'country': '네덜란드'}
]

result = []
# input_list 에 대한 결과 반환
#### result = llm_chain.apply(input_list)

# 반복문으로 결과 출력
for res in result:
    print(res['text'].strip())

'''
답변 : 
호주의 수도는 캔버라입니다.
중국의 수도는 베이징(北京)입니다.
네덜란드의 수도는 암스테르담입니다.
'''

####################################################################
# 3. generate() : 문자열 대신에 LLMResult를 반환하는 점을 제외하고는 apply와 유사합니다.
# tLLMResult는 토큰 사용량과 종료 이유와 같은 유용한 생성 정보를 자주 포함하고 있습니다
####################################################################

# input_list 에 대한 결과 반환
#### generated_result = llm_chain.generate(input_list)
#### print(generated_result)

'''
결과 : 
generations=[[ChatGeneration(text='호주의 수도는 캔버라입니다.', generation_info={'finish_reason': 'stop', 'logprobs': None}, message=AIMessage(content='호주의 수도는 캔버라입니다.', response_metadata={'finish_reason': 'stop', 'logprobs': None}))], [ChatGeneration(text='중국의 수도는 베이징(北京)입니다.', generation_info={'finish_reason': 'stop', 'logprobs': None}, message=AIMessage(content='중국의 수도는 베이징(北京)입니다.', response_metadata={'finish_reason': 'stop', 'logprobs': None}))], [ChatGeneration(text='네덜란드의 수도는 암스테르담입니다.', generation_info={'finish_reason': 'stop', 'logprobs': None}, message=AIMessage(content='네덜란드의 수도는 암스테르담입니다.', response_metadata={'finish_reason': 'stop', 'logprobs': None}))]] llm_output={'token_usage': {'completion_tokens': 53, 'prompt_tokens': 58, 'total_tokens': 111}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8'} run=[RunInfo(run_id=UUID('bba69988-75c2-4ed9-bb37-61fe9e66d3a9')), RunInfo(run_id=UUID('c1f9382f-8020-4b88-ab3c-bcb08b97782f')), RunInfo(run_id=UUID('178ce31e-07ec-45ae-85b4-e75a1769601e'))]
'''

# 토큰 사용량 출력
#generated_result.llm_output
#### print(generated_result.llm_output)
''' 
결과 : 
{'token_usage': {'completion_tokens': 53, 'prompt_tokens': 58, 'total_tokens': 111}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_b28b39ffa8'}
'''


# run ID 출력
#generated_result.run
#### print(generated_result.run)
''' 
[RunInfo(run_id=UUID('61746293-1640-4e5b-87c6-49c90144fb3a')), RunInfo(run_id=UUID('58d94968-b188-42e5-954a-0cca61f9cd40')), RunInfo(run_id=UUID('c573eaed-1842-47d8-9a05-a14aaa40514d'))]
'''

# 답변 출력
#### for gen in generated_result.generations:
####     print(gen[0].text.strip())

''' 
호주의 수도는 캔버라입니다.
중국의 수도는 베이징(北京)입니다.
네덜란드의 수도는 암스테르담입니다.
'''    

####################################################################
# 4. 2개 이상의 변수를 템플릿 안에 정의
# 2개 이상의 변수를 적용하여 템플릿을 생성할 수 있습니다.
# 2개 이상의 변수(input_variables) 를 활용하여 템플릿 구성
####################################################################

# 질문 템플릿 형식 정의
template = '{area1} 와 {area2} 의 시차는 몇시간이야?'

# 템플릿 완성
prompt = PromptTemplate(template=template, input_variables=['area1', 'area2'])

# 연결된 체인(Chain)객체 생성
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 체인 실행: run() 
#print(llm_chain.run(area1='서울', area2='파리'))
#### print(llm_chain.invoke({'area1': '서울', 'area2': '파리'}))

''' 
결과 :
{'area1': '서울', 'area2': '파리', 'text': '서울과 파리의 시차는 8시간입니다. 서울은 GMT+9 시간대에 속하고, 파리는 GMT+1 시간대에 속하기 때문에 시차가 8시간이 발생합니다.'}
'''

input_list = [
    {'area1': '파리', 'area2': '뉴욕'},
    {'area1': '서울', 'area2': '하와이'},
    {'area1': '켄버라', 'area2': '베이징'}
]

# 반복문으로 결과 출력
result = llm_chain.apply(input_list)
print(result)
''' 
[{'text': '파리와 뉴욕의 시차는 6시간입니다. 파리는 그리니치 평균시(GMT+1)를 따르고, 뉴욕은 동부 표준시(EST)를 따르기 때문에 시차가 발생합니다.'}, {'text': '서울과 하와이의 시차는 19시간입니다. 서울은 GMT+9 시간대에 속해 있고, 하와이는 GMT-10 시간대에 속해 있기 때문입니다.'}, {'text': '켄버라와 베이징의 시차는 2시간입니다. 켄버라는 GMT+10 시간대에 속하고, 베이징은 GMT+8 시간대에 속하기 때문입니다.'}]
''' 
for res in result:
    print(res['text'].strip())

''' 
파리와 뉴욕의 시차는 6시간입니다. 파리는 그리니치 평균시(GMT+1)를 따르고, 뉴욕은 동부 표준시(EST)를 따르기 때문에 시차가 발생합니다.
서울과 하와이의 시차는 19시간입니다. 서울은 GMT+9 시간대에 속해 있고, 하와이는 GMT-10 시간대에 속해 있기 때문입니다.
켄버라와 베이징의 시차는 2시간입니다. 켄버라는 GMT+10 시간대에 속하고, 베이징은 GMT+8 시간대에 속하기 때문입니다.
''' 

####################################################################
# 5. 스트리밍(streaming)
# 스트리밍 옵션은 질의에 대한 답변을 실시간으로 받을 때 유용합니다.
# 다음과 같이 streaming=True 로 설정하고 스트리밍으로 답변을 받기 위한 StreamingStdOutCallbackHandler() 을 콜백으로 지정합니다.
####################################################################
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 객체 생성
llm = ChatOpenAI(temperature=0,               # 창의성 (0.0 ~ 2.0) 
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-3.5-turbo',  # 모델명
                 streaming=True,              
                 callbacks=[StreamingStdOutCallbackHandler()]
                )

# 질의내용
question = '대한민국의 수도는 뭐야?'

# 스트리밍으로 답변 출력
#response = llm.predict(question)
response = llm.invoke(question)
print(response)

''' 
대한민국의 수도는 서울이야.content='대한민국의 수도는 서울이야.' response_metadata={'finish_reason': 'stop'}
'''