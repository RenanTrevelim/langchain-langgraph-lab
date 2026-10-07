# ============================
# CONFIGURAÇÃO E TIPOS
# ============================
import os
import operator
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# ============================
# VALIDAÇÃO COM PYDANTIC
# ============================
from pydantic import BaseModel, Field, field_validator, ValidationError

# ============================
# MODELO E PROMPTS
# ============================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================
# DOCUMENTOS, EMBEDDINGS E RAG
# ============================
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================
# GRAFO
# ============================
from langgraph.graph import StateGraph, START, END


# ============================
# 1. DADOS DO PEDIDO
# ============================

class SolicitacaoEstudo(BaseModel):
    """Organiza as informações fornecidas pelo aluno."""

    objetivo : str | None = Field(
        default= None,
        min_length=1,
        description=(
            "O que o aluno QUER APRENDER. "
            "Exemplo: 'quero aprender Python' -> 'aprender Python'. "
            "Use null somente se não houver objetivo informado"
        )
    )

    conhecimento : str | None = Field(
        default= None,
        description=(
            "Lista do que o aluno JÁ SABE. "
            "Exemplo: 'Já conheço Java e C++' -> ['Java', 'C++']. "
            "Não inclua algo apenas porque o aluno quer aprender. "
            "Use [] se disser que começa do zero e null se não informar."
        )
    )

    horas_por_dia_min: float | None = Field(
        default=None,
        gt=0,
        le=24,
        description=(
            "Mínimo de horas disponíveis por dia. "
            "Para '3 a 4 horas por dia', retorne 3. "
            "Para '3 horas por dia', retorne 3."
        )
    )

    horas_por_dia_max: float | None = Field(
        default=None,
        gt=0,
        le=24,
        description=(
            "Máximo de horas disponíveis por dia. "
            "Para '3 a 4 horas por dia', retorne 4. "
            "Para '3 horas por dia', retorne 3."
        )
    )

    dias_por_semana: int | None = Field(
        default=None,
        ge=1,
        le=7,
        description=(
            "Quantidade de dias de estudo por semana. "
            "'Todos os dias' significa 7. "
            "'De segunda a sexta' ou 'dias úteis' significa 5. "
            "Se não estiver claro, retorne null."
        )
)

    semanas : int | None = Field(
        default= None,
        gt=0,
        description="Quantidade de semanas para estudar. Se não informar, use null."
    )

# ============================
# 2. CRIANDO O MODELO
# ============================

def criar_modelo():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configurar GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# ============================
# 3. CRIAR A CHAIN DA EXECUÇÃO
# ============================

def criar_extrator(modelo):
    prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """
        Extraia as informações do pedido de estudo.

        Diferencie:
        - objetivo: o que o aluno quer aprender.
        - conhecimentos: o que o aluno já sabe.

        Regras:
        - Um assunto desejado não é um conhecimento prévio.
        - O aluno pode já conhecer um assunto e querer aprofundá-lo.
          Nesse caso, ele pode aparecer nos dois campos.
        - Entenda erros simples de digitação, como "possu" por "possuo".
        - Preserve todos os conhecimentos mencionados.
        - Para um valor diário único, preencha mínimo e máximo
          com o mesmo valor.
        - Para um intervalo diário, preserve os dois extremos.
        - Não invente dias por semana ou duração do plano.
        - Use null para informações ausentes.
        - Se disser que não possui conhecimentos, use uma lista vazia.

        Exemplo:
        Pedido: Quero aprender SQL, já conheço Excel e tenho
        1 hora por dia para estudar.

        Extração esperada:
        objetivo: aprender SQL
        conhecimentos: lista contendo Excel
        horas_por_dia_min: 1
        horas_por_dia_max: 1
        dias_por_semana: null
        semanas: null
        """
        ),
        ("human", "{pedido}")
    ])

    modelo_estruturado = modelo.with_structured_output(SolicitacaoEstudo)

    chain = prompt | modelo_estruturado

    return chain

# ============================
# VERIFICAR INFORMÇÕES AUSENTES
# ============================

def verificar_pendencias(solicitacao):
    """Cria perguntas somente sobre os dados que faltam."""

    pendencias = []

    if solicitacao.objetivo is None:
        pendencias.append(
            "O que você gostaria de aprender?"
        )

    if solicitacao.conhecimento is None:
        pendencias.append(
            "Quais conhecimentos você já possui? Pode dizer se começa do zero."
        )

    if (
        solicitacao.horas_por_dia_min is None or solicitacao.horas_por_dia_max is None
    ):
        pendencias.append(
            "Quantas horas por dia você pode estudar?"
        )

    if solicitacao.dias_por_semana is None:
        pendencias.append(
            "Quantos dias por semana você pretende estudar?"
        )

    if solicitacao.semanas is None:
        pendencias.append(
            "Por quantas semanas você quer organizar o plano?"
        )

    return pendencias

# ============================
# EXECUTAR PRIMEIRO TESTE
# ============================

def main():
    modelo = criar_modelo()
    extrator = criar_extrator(modelo)

    pedido = input("[ALUNO]: ").strip()

    if not pedido:
        print("Digite seu pedido de estudo...")
        return

    solicitacao = extrator.invoke({"pedido": pedido})

    print("\n=== DADOS EXTRAIDOS ===")
    print(f"Objetivo: {solicitacao.objetivo}")
    print(f"Conhecimentos: {solicitacao.conhecimento}")
    print(f"Horas diárias mínimas: {solicitacao.horas_por_dia_min}")
    print(f"Horas diárias máximas: {solicitacao.horas_por_dia_max}")
    print(f"Dias por semana: {solicitacao.dias_por_semana}")
    print(f"Semanas: {solicitacao.semanas}")

    pendencias = verificar_pendencias(solicitacao)

    if pendencias:
        print("\n[PRECISAMOS DE MAIS INFORMAÇÕES]")

        for pergunta in pendencias:
            print(f"- {pergunta}")

        return
    

if __name__ == "__main__":
    main()