import httpx
from loguru import logger

from langflow.custom import Component
from langflow.helpers.data import data_to_text
from langflow.io import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output,
    SecretStrInput,
)
from langflow.schema import Data
from langflow.schema.message import Message


class TavilySearchComponent(Component):
    display_name = "Busca AI Tavily"
    description = (
        "**Tavily AI** é um mecanismo de busca otimizado para LLMs e RAG, "
        "voltado para resultados eficientes, rápidos e persistentes."
    )
    icon = "TavilyIcon"

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
            info="A consulta de busca que será executada no Tavily.",
            tool_mode=True,
        ),
        DropdownInput(
            name="search_depth",
            display_name="Profundidade da Busca",
            info="A profundidade da busca.",
            options=["básica", "avançada"],
            value="avançada",
            advanced=True,
        ),
        DropdownInput(
            name="topic",
            display_name="Tópico da Busca",
            info="A categoria da busca.",
            options=["geral", "notícias"],
            value="geral",
            advanced=True,
        ),
        DropdownInput(
            name="time_range",
            display_name="Intervalo de Tempo",
            info="Período retroativo a partir da data atual para incluir nos resultados da busca.",
            options=["dia", "semana", "mês", "ano"],
            value=None,
            advanced=True,
            combobox=True,
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
            info="Incluir lista de imagens relacionadas à consulta na resposta.",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="include_answer",
            display_name="Incluir Resposta",
            info="Incluir uma breve resposta para a consulta original.",
            value=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Dados", name="data", method="fetch_content"),
        Output(display_name="Texto", name="text", method="fetch_content_text"),
    ]

    def fetch_content(self) -> list[Data]:
        try:
            url = "https://api.tavily.com/search"
            headers = {"content-type": "application/json", "accept": "application/json"}
            payload = {
                "api_key": self.api_key,
                "query": self.query,
                "search_depth": self.search_depth,
                "topic": self.topic,
                "max_results": self.max_results,
                "include_images": self.include_images,
                "include_answer": self.include_answer,
                "time_range": self.time_range,
            }

            with httpx.Client() as client:
                response = client.post(url, json=payload, headers=headers)

            response.raise_for_status()
            search_results = response.json()

            data_results: list[Data] = []

            if self.include_answer and search_results.get("answer"):
                data_results.append(Data(text=search_results["answer"]))

            for result in search_results.get("results", []):
                content = result.get("content", "")
                data_results.append(
                    Data(
                        text=content,
                        data={
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "content": content,
                            "score": result.get("score"),
                        },
                    )
                )

            if self.include_images and search_results.get("images"):
                data_results.append(
                    Data(text="Imagens encontradas", data={"images": search_results["images"]})
                )

        except httpx.HTTPStatusError as exc:
            error_message = f"Erro HTTP ocorrido: {exc.response.status_code} - {exc.response.text}"
            logger.error(error_message)
            return [Data(text=error_message, data={"error": error_message})]
        except httpx.RequestError as exc:
            error_message = f"Erro de requisição ocorrido: {exc}"
            logger.error(error_message)
            return [Data(text=error_message, data={"error": error_message})]
        except ValueError as exc:
            error_message = f"Formato de resposta inválido: {exc}"
            logger.error(error_message)
            return [Data(text=error_message, data={"error": error_message})]
        else:
            self.status = data_results  # tipo: ignore
            return data_results

    def fetch_content_text(self) -> Message:
        data = self.fetch_content()
        result_string = data_to_text("{text}", data)
        self.status = result_string
        return Message(text=result_string)
