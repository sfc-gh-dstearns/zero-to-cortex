import streamlit as st # Import python packages
from snowflake.snowpark.context import get_active_session
session = get_active_session() # Get the current credentials

import pandas as pd

pd.set_option("max_colwidth",None)

### Default Values
# context = st.session_state.context # Num-chunks provided as context. Play with this to check how it affects your accuracy
slide_window = 7 # how many last conversations to remember. This is the slide window.
#debug = 1 #Set this to 1 if you want to see what is the text created as summary and sent to get chunks
#use_chat_history = 0 #Use the chat history by default

### Functions

def main():
    
    st.title(f":speech_balloon: Snowflake Cortex Chatbot for Enterprise Data")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://miro.medium.com/v2/resize:fit:1200/1*HI4Kj_HQ-JYQHlLD-KSkDw.jpeg")
    with col2:
        st.image("https://www.snowflake.com/wp-content/uploads/2024/04/Snowflake_ARCTIC_4-24_Blog_hero.png")

    config_options()
    init_messages()
     
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with st.chat_message(name = "User", avatar="https://cdn2.iconfinder.com/data/icons/men-women-from-all-over-the-world-1/92/man-woman-people-person-avatar-face-user_64-512.png"):
        st.write(f"""Hi, I am a world class sommelier :wine_glass: based on the {st.session_state.model_name} model. I have a large list of wines you can explore.
                        I can help you find a wine that suits your palate, budget, varietal needs, or regional preferences. If you don't like my 
                            recommendations, feel free to change the model type. Pay special attention to the **CONTEXT WINDOW** slider in the
                            sidebar. This can help provide me with extra context to answer your questions. Additionally,
                            I will retain some memory of our conversation for a better chat experience until you click "Start Over".""")
    # Accept user input
    if question := st.chat_input("How can I help?", key="question_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
    
            question = question.replace("'","")
    
            with st.spinner(f"{st.session_state.model_name} thinking..."):
                response = complete(question)
                res_text = response[0].RESPONSE     
            
                res_text = res_text.replace("'", "")
                message_placeholder.markdown(res_text)
        
        st.session_state.messages.append({"role": "assistant", "content": res_text})

def config_options():
    st.sidebar.selectbox('Select your model:',(
                                            'snowflake-arctic',
                                            'mistral-large',
                                            'mistral-7b',
                                            'mixtral-8x7b',
                                            'llama3-70b',
                                            'llama3-8b'
                                           'llama2-70b-chat',
                                           'reka-flash',
                                           'reka-core',
                                           'gemma-7b'), key="model_name")
    st.sidebar.slider(label="Context Window", min_value=5, max_value=50, key="context")
    # For educational purposes. Users can chech the difference when using memory or not
    st.sidebar.checkbox('Do you want me to remember the chat history?', key="use_chat_history", value = True)

    st.sidebar.checkbox('Debug: Click to see summary generated of previous conversation', key="debug", value = True)
    st.sidebar.button("Start Over", key="clear_conversation")
    st.sidebar.expander("Session State").write(st.session_state)


def init_messages():

    # Initialize chat history
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []

    
def get_similar_chunks (question):
    cmd = """
        with results as
        (
            SELECT
                VECTOR_COSINE_SIMILARITY(information_embeds,
                    snowflake.cortex.embed_text_768('snowflake-arctic-embed-m', ?)) as distance,
                full_description
        from z2c.cortex.wine_reviews
        order by distance desc
        limit ?)
        select full_description from results 
    """
    df_chunks = session.sql(cmd, params=[question, st.session_state.context]).to_pandas()       

    df_chunks_length = len(df_chunks) -1

    similar_chunks = ""
    for i in range (0, df_chunks_length):
        similar_chunks += df_chunks._get_value(i, 'FULL_DESCRIPTION')

    similar_chunks = similar_chunks.replace("'", "")
             
    return similar_chunks


def get_chat_history():
#Get the history from the st.session_stage.messages according to the slide window parameter
    
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history

    
def summarize_question_with_history(chat_history, question):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    summary = df_response[0].RESPONSE     

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(summary)

    summary = summary.replace("'", "")

    return summary

def create_prompt (myquestion):

    if st.session_state.use_chat_history:
        chat_history = get_chat_history()

        if chat_history != "": #There is chat_history, so not first question
            question_summary = summarize_question_with_history(chat_history, myquestion)
            prompt_context =  get_similar_chunks(question_summary)
            # st.write(prompt_context)
        else:
            prompt_context = get_similar_chunks(myquestion) #First question when using history
    else:
        prompt_context = get_similar_chunks(myquestion)
        chat_history = ""
    prompt = f"""You are a world class sommelier. Our guests seek knowledge and guidance. Do not hallucinate, the guests don't like that.
                If you don't know what to recommend just say so. Please answer the questions based on the provided 
                context found between the tags <context> and </context>. The guest's question will be between the
                <question> and </question> tags. Please present the wine nicely. Explain where it is from, the variety of wine it is, and the price.
                You offer a chat experience considering the information included in the CHAT HISTORY
                provided between <chat_history> and </chat_history> tags..
                When answering the question contained between <question> and </question> tags
                be concise and do not hallucinate. 
                If you don´t have the information just say so.
           
                Do not mention the CONTEXT used in your answer.
                Do not mention the CHAT HISTORY used in your asnwer.
           
               <chat_history>
               {chat_history}
               </chat_history>
               <context>          
               {prompt_context}
               </context>
               <question>  
               {myquestion}
               </question>
               Answer: 
           """

    return prompt


def complete(myquestion):

    prompt = create_prompt (myquestion)
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """
    
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    return df_response

if __name__ == "__main__":
    main()
