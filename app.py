import os
import platform
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Configuración inicial de la página
st.set_page_config(
    page_title="Agente RAG - Chat PDF", page_icon="💬", layout="centered"
)

# App title and presentation
st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.caption(f"Versión de Python: {platform.python_version()}")

# Sidebar information & API Key input
with st.sidebar:
  st.header("⚙️ Configuración")
  st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

  st.markdown("---")

  # Get API key from user
  ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")
  if ke:
    os.environ["OPENAI_API_KEY"] = ke
    st.success("API Key configurada correctamente", icon="🔑")
  else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Layout de presentación (Imagen + Carga de archivo)
col_img, col_info = st.columns([1, 1])

with col_img:
  # Load and display image
  try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=320)
  except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

with col_info:
  st.markdown("### 📄 Cargar Documento")
  # PDF uploader
  pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

st.markdown("---")

# Process the PDF if uploaded
if pdf is not None and ke:
  try:
    # Extract text from PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    st.info(f"Texto extraído: {len(text)} caracteres")

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=20, length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.success(f"Documento dividido en {len(chunks)} fragmentos")

    # Create embeddings and knowledge base
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # User question interface
    st.subheader("❓ Consulta el documento")
    user_question = st.text_area(
        "Escribe qué quieres saber sobre el documento",
        placeholder="Ej. ¿Cuál es el tema principal de este archivo?",
    )

    # Process question when submitted
    if user_question:
      docs = knowledge_base.similarity_search(user_question)

      # Use a current model instead of deprecated text-davinci-003
      llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")

      # Load QA chain
      chain = load_qa_chain(llm, chain_type="stuff")

      # Run the chain
      response = chain.run(input_documents=docs, question=user_question)

      # Display the response
      st.markdown("### 💡 Respuesta:")
      st.success(response)

  except Exception as e:
    st.error(f"Error al procesar el PDF: {str(e)}")
    import traceback

    st.error(traceback.format_exc())
elif pdf is not None and not ke:
  st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
  st.info("Por favor carga un archivo PDF para comenzar")
