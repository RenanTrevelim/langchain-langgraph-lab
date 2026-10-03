import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

# 1. Criando Classificador do Atendimento

class ClassificacaoAtendimento(BaseModel):

    setor: Literal["financeiro", "tecnico", "comercial"] = Field(
        description="Identificar de qual setor o comentário está se referindo."
    )

    resumo: str = Field(
        min_length=1,
        description="Vai dar uma breve explicação do problema, referente ao setor que foi identificado"
    )

    @field_validator("setor", mode="before")
    @classmethod
    def normalizar_setor(cls, valor):
        # EX: "Técnico -> tecnico"
        if isinstance(valor, str):
            valor = valor.strip().lower().replace("é", "e")
            return valor

        return valor

# 2. Criando o Modelo

def criar_modelo():
    """Carrega o Modelo e confiure o GROQ"""

    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Configure GROQ_API_KEY no arquivo .env")

    modelo = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    return modelo

# 3. Criando modelo Classificador

def modelo_classificador(modelo):
    """Cria um modelo que classifica com base no Atendimento"""

    prompt = ChatPromptTemplate.from_template(
        """
        Classifique a solicitação do cliente em um destets setores:

        - financeiro: cobranças, pagamentos. boletos e estornos.
        - tecnico: problemas no funcionamento ou uso de produtos.
        - comercial: dúvidas sobre compras e escolha de produtos.

        Se houver mais de um assunto, escolha o setor da solicitação principal.
        Também faça um resumo curto do que o cliente precisa.

        Solicitação: {pergunta}
        """
    )

    llm_classificador = modelo.with_structured_output(ClassificacaoAtendimento)

    return prompt | llm_classificador

# 4. Montando Prompts

def criando_chains_setor(modelo):
    """Montando os prompts por setor"""

    prompt_financeiro = ChatPromptTemplate.from_template(
        """
        Você atende no setor financeiro de uma loja
        Oriente o cliente de forma simples, breve e cordial.

        Você não tem acesso aos dados de cobrança.
        Não diga que consultou pagamentos ou realizou estornos.
        Quando necessário, indique quais informações o cliente
        deverá fornecer ao atendimento oficial.

        Solicitação: {pergunta}
        """
    )

    prompt_comercial = ChatPromptTemplate.from_template(
        """
        Você atende no setor comercial de uma loja.
        Ajude o cliente a entender qual produto atende à necessidade dele.
        Pergunte sobre uso e orçamento quando necessário.

        Não invente preços, promoções ou disponibilidade de estoque.
        Responda de forma breve e cordial.

        Solicitação: {pergunta}
        """
    )

    prompt_tecnico = ChatPromptTemplate.from_template(
        """
        Você atende no suporte técnico de uma loja.
        Oriente o cliente com verificações simples e seguras.
        Se faltarem informações sobre o produto ou problema,
        faça uma pergunta para entender melhor.

        Responda de forma breve e cordial.

        Solicitação: {pergunta}
        """
    )

    chain_finaneiro = prompt_financeiro | modelo | StrOutputParser()
    chain_tecnico = prompt_tecnico | modelo | StrOutputParser()
    chain_comercial = prompt_comercial | modelo | StrOutputParser()

    return {
        "financeiro": chain_finaneiro,
        "tecnico": chain_tecnico,
        "comercial": chain_comercial
    }


# =============================
# 5. ENCAMINHAR AO SETOR
# =============================

def rotear_consulta(setor, pergunta, chains):
    """Escolher a chain do setor e envia a pergunta original"""

    chain_escolhida = chains[setor]

    resposta = chain_escolhida.invoke({
        "pergunta": pergunta
    })

    return resposta


# =============================
# 6. EXECUTAR O PROGRAMA
# =============================

def main():
    # Prepara os componentes
    modelo = criar_modelo()
    classificador = modelo_classificador(modelo)

    # usamos o modelo normal para responder ao cliente
    chains = criando_chains_setor(modelo)

    pergunta = input("[CLIENTE]: ").strip()

    if not pergunta:
        print("Digite uma solicitação para continuar...")
        return

    # Primeiro chamada ao modelo: classificar
    classificacao = classificador.invoke({
        "pergunta": pergunta
    })

    print(f"\nSetor -> {classificacao.setor}")

    # Segunda chamada ao modelo: atender no setor escolhido
    resposta = rotear_consulta(classificacao.setor, pergunta, chains)

    print(f"\n[ATENDIMENTO] : {resposta}")


if __name__ == "__main__":
    main()
    

