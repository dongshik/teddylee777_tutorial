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
model_list = sorted([m.id for m in client.models.list().data])
for m in model_list:
    print(m)

####################################################################
# ChatOpenAI
# OpenAI 사의 채팅 전용 Large Language Model(llm)
# temperature : 사용할 샘플링 온도는 0과 2 사이에서 선택합니다. 0.8과 같은 높은 값은 출력을 더 무작위하게 만들고, 0.2와 같은 낮은 값은 출력을 더 집중되고 결정론적으로 만듭니다.
# max_tokens : 채팅 완성에서 생성할 토큰의 최대 개수입니다.
####################################################################