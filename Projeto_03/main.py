import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document

# Localiza a pasta "documentos" ao lado deste main.py.
PASTA_DOCUMENTOS = Path(__file__).resolve().parent / "documentos"

# 1. Criando o Modelo

def criar_modelo():
    """Carrega o modelo e configura o GROQ"""

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configure o GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# 2. Lendo os Documentos

def carregar_documentos():
    """Lê os arquivos .txt e retorna uma lista de Documents."""

    # Encontra todos arquivos terminados em .txt
    arquivos = sorted(PASTA_DOCUMENTOS.glob("*.txt"))

    if not arquivos:
        print(f"Nenhum arquivo .txt encontrado em {PASTA_DOCUMENTOS}")

    documentos = []

    # Lê um arquivo por vez
    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8").strip()

        if not texto:
            print(f"Arquivo {arquivo.name} encontra-se vazio")

        documento = Document(
            page_content=texto,
            metadata={"fonte": arquivo.name}
        )

        documentos.append(documento)

    return documentos

# 3. Dividir Documentos

def dividir_documentos(documentos):
    """Pega os documentos e faz a divisão lendo os trechos"""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100 
    )

    trechos = divisor.split_documents(documentos)

    print(f"Documentos carregados: {len(documentos)}")
    print(f"Trechos Criados: {len(trechos)}")

    return trechos

# 4. Criando Banco + Embeddings

def cria_banco_embeddings(trechos):
    """Criando Banco Vetorial e os embeddings que usaremos nele."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    banco_vetorial = FAISS.from_documents(
        documents=trechos,
        embedding=embeddings
    )

    return banco_vetorial

# 5. Busca pelo Contexto

def buscar_contexto(banco, pergunta):
    """Buscando o contexto por similiariedade dos textos."""

    documentos = banco.similarity_search(
        pergunta,
        k=3
    )

    return documentos

# 6. Gerando a resposta

def gerar_resposta(pergunta, contexto, modelo):
    """Gerando as respostas com base no contexto e modelo"""

    prompt = PromptTemplate.from_template(
        "Você é um assistente de uma loja. \n"
        "Responda com base no contexto que foi passado, não invente nada que não esteja no contexto. \n"
        "Se alguma informação não estiver no contexto, diga que não encontrou essa informação. \n"
        "Contexto: {contexto}\n"
        "Pergunta: {pergunta}\n\n"
        "Resposta"
    )

    chain = prompt | modelo | StrOutputParser()

    resposta = chain.invoke({
        "pergunta": pergunta,
        "contexto": contexto
    })

    return resposta

def main():
    documentos = carregar_documentos()
    trechos = dividir_documentos(documentos)
    banco = cria_banco_embeddings(trechos)

    modelo = criar_modelo()

    pergunta = input("[CLIENTE]: ").strip()

    if not pergunta:
        print("Digite uma pergunta...")
        return

    encontrados = buscar_contexto(banco, pergunta)

    # Mostra o que será enviado ao modelo.
    print("\n=== TRECHOS RECUPERADOS ===")

    for numero, documento in enumerate(encontrados, start=1):
        print(f"\nTrecho {numero}")
        print(f"Fonte: {documento.metadata['fonte']}")
        print(documento.page_content)

    # Junta os trechos recuperados em um único texto.
    contexto = "\n\n".join(documento.page_content for documento in encontrados)

    resposta = gerar_resposta(pergunta, contexto, modelo)

    print(f"\n [RESPOSTA]: {resposta}")

    fontes = sorted({
        documento.metadata["fonte"] for documento in documentos
    })

    print(f"\nFontes consultadas: {', '.join(fontes)}")

if __name__ == "__main__":
    main()