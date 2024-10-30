import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 어떻게 되나요?",
    "모델은 어떤걸 썼나요?",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요?",
    "데이터는 뭘 이용했나요?",
    "어떤 어려움이 있었나요?"
]

answers = [
    "딥보이스(TTS)를 활용한 심리상담 AI챗봇 시스템입니다.",
    "다음 모델을 활용했습니다.  \n  [Speech to Text] Whisper  \n  [AI챗봇] 00000  \n  [Text to Speech] xTTS_v2",
    "프로젝트 팀 구성은 다음과 같습니다. \n 팀장: 유재현 / 팀원: 한수연, 황지영",
    "2024-10-24 ~ 2024-11-18 동안 진행했습니다.",
    "팀장은 유재현입니다.",
    "ai-hub의 데이터를 활용했습니다.",
    "점차 깊어지는 대화 흐름 구현에 어려움이 있었습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("포트폴리오 Q&A 챗봇")
st.write("포트폴리오에 관한 질문을 입력해보세요. 예: 주제가 어떻게 되나요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")
