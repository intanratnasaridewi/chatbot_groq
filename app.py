
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os
os.environ['GROQ_API_KEY']=st.secrets['groq_api']


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    Anda adalah seorang analis data di sebuah perusahaan. Anda sedang berinteraksi dengan seorang pengguna yang menanyakan pertanyaan tentang database perusahaan. 
    Berdasarkan skema tabel di bawah ini, tuliskan query SQL yang akan menjawab pertanyaan pengguna. Perhatikan juga riwayat percakapan dalam menyusun query.

    <SCHEMA>{schema}</SCHEMA>
    
    Riwayat Percakapan: {chat_history}
    
    Tuliskan hanya query SQL-nya saja dan tidak ada teks lain. Jangan membungkus query SQL dengan teks tambahan, bahkan backticks sekalipun.
    
    Contoh:
    Pertanyaan: Pegawai yang memiliki status pegawai asn?
    SQL Query: SELECT * FROM public.tb_pegawai WHERE status_pegawai = 'asn';
    Pertanyaan:Bagaimana proporsi bidang pegawai berdasarkan tabel struktur organisasi?
    SQL Query: SELECT Department,(EmployeeCount * 100.0 / (SELECT SUM(EmployeeCount) FROM StrukturOrganisasi)) AS ProportionPercentage FROM StrukturOrganisasi ORDER BY ProportionPercentage DESC;
    Pertanyaan:Apa pendidikan terakhir yang paling banyak dimiliki pegawai?
    SQL Query:SELECT HighestEducation, COUNT(*) AS Total FROM Pendidikan GROUP BY HighestEducation ORDER BY Total DESC LIMIT 1;

    
    Sekarang giliran Anda:
    
    Pertanyaan: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    Anda adalah seorang analis data di sebuah perusahaan. Anda sedang berinteraksi dengan seorang pengguna yang menanyakan pertanyaan tentang database perusahaan.
    Berdasarkan skema tabel di bawah ini, pertanyaan, query SQL, dan respons SQL, tuliskan respons dalam bahasa alami.
    <SCHEMA>{schema}</SCHEMA>

    Riwayat Percakapan: {chat_history}
    Query SQL: <SQL>{query}</SQL>
    Pertanyaan Pengguna: {question}
    Respons SQL: {response}
    Respons dalam Bahasa Alami:
    """
  
  prompt = ChatPromptTemplate.from_template(template)
  
  # llm = ChatOpenAI(model="gpt-4-0125-preview")
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! Aku Asistenmu!"),
    ]

st.set_page_config(page_title="Warehouse Chat", page_icon=":speech_balloon:")

st.title("Warehouse Chat")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="xxxx", key="Port")
    st.text_input("User", value="xxxxx", key="User")
    st.text_input("Password", type="password", value="xxxxxx", key="Password")
    st.text_input("Database", value="xxxxx", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))
