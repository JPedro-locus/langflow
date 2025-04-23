from enum import Enum

import httpx
from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
from loguru import logger
from pydantic import BaseModel, Field

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.field_typing import Tool
from langflow.inputs import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    SecretStrInput,
)
from langflow.schema import Data


class TavilySearchDepth(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"


class TavilySearchTopic(Enum):
    GENERAL = "general"
    NEWS = "news"


class TavilySearchSchema(BaseModel):
    query: str = Field(..., description="A consulta de busca que você deseja executar com o Tavily.")
    search_depth: TavilySearchDepth = Field(
        TavilySearchDepth.BASIC,
        description="A profundidade da busca.",
    )
    topic: TavilySearchTopic = Field(
        TavilySearchTopic.GENERAL,
        description="A categoria da busca.",
    )
    max_results: int = Field(
        5,
        description="O número máximo de resultados de busca a retornar.",
    )
    include_images: bool = Field(
        False,
        description="Incluir uma lista de imagens relacionadas à consulta na resposta.",
    )
    include_answer: bool = Field(
        False,
        description="Incluir uma breve resposta para a consulta original.",
    )


class TavilySearchToolComponent(LCToolComponent):
    display_name = "Busca AI Tavily [OBSOLETO]"
    description = (
        "**Tavily AI** é um mecanismo de busca otimizado para LLMs e RAG, "
        "voltado para resultados eficientes, rápidos e persistentes. Pode ser usado "
        "de forma independente ou como ferramenta de agente.\n\n"
        "Observação: verifique a opção 'Avançado' para mais configurações."
    )
    icon = "TavilyIcon"
    name = "TavilyAISearch"
    documentation = "https://docs.tavily.com/"
    legacy = True

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="Chave de API Tavily",
            required=True,
            info="Sua chave de API do Tavily.",
        ),
        MessageTextInput(
            name="query",
            display_name="Consulta de Busca",
            info="A consulta de busca que você deseja executar com o Tavily.",
        ),
        DropdownInput(
            name="search_depth",
            display_name="Profundidade da Busca",
            info="A profundidade da busca.",
            options=list(TavilySearchDepth),
            value=TavilySearchDepth.ADVANCED,
            advanced=True,
        ),
        DropdownInput(
            name="topic",
            display_name="Tópico da Busca",
            info="A categoria da busca.",
            options=list(TavilySearchTopic),
            value=TavilySearchTopic.GENERAL,
            advanced=True,
        ),
        IntInput(
            name="max_results",
            display_name="Resultados Máximos",
            info="O número máximo de resultados de busca a retornar.",
            value=5,
            advanced=True,
        ),
        BoolInput(
            name="include_images",
            display_name="Incluir Imagens",
            info="Incluir uma lista de imagens relacionadas à consulta na resposta.",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="include_answer",
            display_name="Incluir Resposta",
            info="Incluir uma breve resposta para a consulta original.",
            value=False,
            advanced=True,
        ),
    ]

    def run_model(self) -> list[Data]:
        # Converte valores de string para instâncias de enum com validação
        try:
            search_depth_enum = (
                self.search_depth
                if isinstance(self.search_depth, TavilySearchDepth)
                else TavilySearchDepth(str(self.search_depth).lower())
            )
        except ValueError as e:
            error_message = f"Valor de profundidade de busca inválido: {e!s}"
            self.status = error_message
            return [Data(data={"error": error_message})]

        try:
            topic_enum = (
                self.topic
                if isinstance(self.topic, TavilySearchTopic)
                else TavilySearchTopic(str(self.topic).lower())
            )
        except ValueError as e:
            error_message = f"Valor de tópico inválido: {e!s}"
            self.status = error_message
            return [Data(data={"error": error_message})]

        return self._tavily_search(
            self.query,
            search_depth=search_depth_enum,
            topic=topic_enum,
            max_results=self.max_results,
            include_images=self.include_images,
            include_answer=self.include_answer,
        )

    def build_tool(self) -> Tool:
        # Constrói a ferramenta estruturada para uso com LangChain
        return StructuredTool.from_function(
            name="tavily_search",
            description="Executa uma busca web usando a API do Tavily.",
            func=self._tavily_search,
            args_schema=TavilySearchSchema,
        )

    def _tavily_search(
        self,
        query: str,
        *,
        search_depth: TavilySearchDepth = TavilySearchDepth.BASIC,
        topic: TavilySearchTopic = TavilySearchTopic.GENERAL,
        max_results: int = 5,
        include_images: bool = False,
        include_answer: bool = False,
    ) -> list[Data]:
        # Valida tipos de enum
        if not isinstance(search_depth, TavilySearchDepth):
            msg = f"Profundidade de busca inválida: {search_depth}"
            raise TypeError(msg)
        if not isinstance(topic, TavilySearchTopic):
            msg = f"Tópico de busca inválido: {topic}"
            raise TypeError(msg)

        try:
            url = "https://api.tavily.com/search"
            headers = {
                "content-type": "application/json",
                "accept": "application/json",
            }
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth.value,
                "topic": topic.value,
                "max_results": max_results,
                "include_images": include_images,
                "include_answer": include_answer,
            }

            # Executa requisição HTTP
            with httpx.Client() as client:
                response = client.post(url, json=payload, headers=headers)

            response.raise_for_status()
            resultados = response.json()

            data_results = [
                Data(
                    data={
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": item.get("content"),
                        "score": item.get("score"),
                    }
                )
                for item in resultados.get("results", [])
            ]

            # Insere resposta resumida, se solicitado
            if include_answer and resultados.get("answer"):
                data_results.insert(0, Data(data={"answer": resultados["answer"]}))

            # Adiciona lista de imagens, se solicitado
            if include_images and resultados.get("images"):
                data_results.append(Data(data={"images": resultados["images"]}))

            self.status = data_results  # armazena o status para logs

        except httpx.HTTPStatusError as e:
            error_message = f"Erro HTTP: {e.response.status_code} - {e.response.text}"
            logger.debug(error_message)
            self.status = error_message
            raise ToolException(error_message) from e
        except Exception as e:
            error_message = f"Erro inesperado: {e}"
            logger.opt(exception=True).debug("Erro ao executar busca Tavily")
            self.status = error_message
            raise ToolException(error_message) from e
        return data_results
