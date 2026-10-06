import os
from typing import Literal
from pathlib import Path
from decimal import Decimal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

CAMINHO_CATALOGO = Path(__file__).resolve().parent / "catalogo.txt"

# 1. Criando Pydantic - Solicitação

class SolicitacaoOrcamento(BaseModel):
    """Define e valida os dados que a ferramenta recebe."""

    produto : str = Field(
        min_length=1,
        description="Nome do produto encontrado no catálogo."
    )

    preco_unitario : Decimal = Field(
        gt=0,
        allow_inf_nan=False,
        description="Preço unitário encontrado no catálogo"
    )

    quantidade : int = Field(
        gt=0,
        description="Quantidade de unidades solicitada pelo cliente."
    )

    forma_pagamento : Literal["avista", "parcelado"] = Field(
        description="Forma de pagamento escolhida pelo cliente"
    )

    @field_validator("forma_pagamento", mode="before")
    def normaliza_pagamento(cls, valor):
        """Normaliza o texto antes da validação"""

        if isinstance(valor, str):
            valor = valor.strip().lower()
            return valor
        return valor

# ============================
# Ferramenta de calculo
# ============================

@tool(args_schema=SolicitacaoOrcamento)
def calcular_orcamento(
    produto: str,
    preco_unitario:Decimal,
    quantidade: int,
    forma_pagamento: str
) -> str:
    """Calcula o orçamento com 10% de desconto à vista ou sem desconta parcelado"""

    subtotal = preco_unitario * quantidade

    if forma_pagamento == "avista":
        total = subtotal * Decimal("0.90")
    else:
        total = subtotal

    return (
        f"Produto: {produto}\n"
        f"Quantidade: {quantidade}\n"
        f"Pagamento: {forma_pagamento}\n"
        f"Valor sem desconto: R$ {subtotal:.2f}\n"
        f"Total a pagar: R$ {total:.2f}"
    )

# ============================
# 2. Lendo os Documentos
# ============================

def carregar_documentos():
    """Lê os arquivos .txt e retorna uma lista de Documentos"""

    texto = CAMINHO_CATALOGO.read_text(
        encoding="utf-8"
    ).strip()

    if not texto:
        raise ValidationError("O catálogo está vazio!")

    return texto

# ============================
# 3. Dividir em Chunks
# ============================

def dividir_textos(texto):
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )

    trechos = divisor.split_text(texto)

    print(f"Trechos criados: {len(trechos)}")

    return trechos

# ============================
# 4. Criando Banco + Embeddings
# ============================

def cria_banco_embeddings(trechos):
    """Criando Banco Vetorial e os embeddings que usarmos nele"""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    banco_vetorial = FAISS.from_texts(
        texts=trechos,
        embedding=embeddings
    )

    return banco_vetorial

# ============================
# 5. Buscando contexto
# ============================

def buscar_contexto(banco, pergunta):
    """Buscando o contexto por similiaridade dos textos."""

    documentos = banco.similarity_search(
        pergunta,
        k=3
    )

    contexto = "\n\n".join(documento.page_content for documento in documentos)

    return contexto

# ============================
# 6. Criando o Modelo
# ============================

def criar_modelo():
    """Criando o modelo do GROQ"""

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configure GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# ============================
# 7. CRIAR O AGENTE
# ============================

def criar_agente(modelo):
    """Conecta o modelo à ferrament de orçamento."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            Você é um atendente de uma loja.

            Use o catálogo abaixo para identificar o produto e seu preço.
            Não invente produtos ou preços.

            Para calcular um orçamento, use a ferramenta calcular_orcamento.

            Regras:
            - Pegue o nome e o preço do produto no catálogo.
            - Pegue a quantidade e a forma de pagamento na mensagem do cliente.
            - Se faltar alguma dessas informações, pergunte ao cliente.
            - Se houver dúvida sobre qual produto ele quer, peça confirmação.
            - À vista corresponde a 'avista'.
            - Parcelado corresponde a 'parcelado'.
            - Apresente o resultado da ferramenta de forma simples.

            Catálogo:
            {contexto}
            """
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    ferramentas = [calcular_orcamento]

    agente = create_tool_calling_agent(
        llm=modelo,
        tools=ferramentas,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agente,
        tools=ferramentas,
        verbose=True,
        max_iterations=5
    )

    return executor

# ============================
# 8. GERAR RESPOSTA
# ============================

def gerar_resposta(modelo, contexto, pergunta):
    """Junta o modelo, o contexto do catálogo e a pergunta"""

    agente = criar_agente(modelo)

    resultado = agente.invoke({
        "contexto": contexto,
        "input": pergunta
    })

    return resultado["output"]

def main():

    texto = carregar_documentos()
    trechos = dividir_textos(texto)

    banco = cria_banco_embeddings(trechos)

    pergunta = input("[CLIENTE]: ")

    contexto = buscar_contexto(banco, pergunta)

    modelo = criar_modelo()

    resposta = gerar_resposta(modelo, contexto, pergunta)

    print(f"\n\n[PERGUNTA] -> {pergunta}")
    print(f"\n[ATENDENTE] -> {resposta}")
 
if __name__ == "__main__":
    main()