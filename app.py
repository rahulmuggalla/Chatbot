import streamlit as st

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    groq_api = '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'
   
    st.title('Chat wit Groq!')
    st.write("Hello! I'm your friendly Groq Chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation")

    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input('System Prompt: ')
    model = st.sidebar.selectbox(
        'Choose a Model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversatioanl_memory_length = st.sidebar.slider('Conversational Memory Length: ', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversatioanl_memory_length, memory_key='chat_history', return_messages=True)

    user_question = st.text_input('Ask a Question: ')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    groq_chat = ChatGroq(
        groq_api_key = groq_api,
        model_name = model
    )

    

    if user_question:

                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content=system_prompt
                        ),

                        MessagesPlaceholder(
                            variable_name='chat_history'
                        ),

                        HumanMessagePromptTemplate.from_template(
                            "{human_input}"
                        )
                    ]
                )

                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    verbose=False,
                    memory=memory
                )

                response = conversation.predict(human_input=user_question)
                message = {'human': user_question, 'AI': response}
                st.session_state.chat_history.append(message)
                st.write(f'Chatbot: {response}')

if __name__ == '__main__':
    main()
