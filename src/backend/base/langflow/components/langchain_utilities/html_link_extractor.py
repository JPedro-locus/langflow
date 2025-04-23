from typing import Any

from langchain_community.graph_vectorstores.extractors import HtmlLinkExtractor, LinkExtractorTransformer
from langchain_core.documents import BaseDocumentTransformer

from langflow.base.document_transformers.model import LCDocumentTransformerComponent
from langflow.inputs import BoolInput, DataInput, StrInput


class HtmlLinkExtractorComponent(LCDocumentTransformerComponent):
    display_name = "Extrator de Links HTML"
    description = "Extrai hyperlinks de conteúdo HTML."
    documentation = (
        "https://python.langchain.com/v0.2/api_reference/community/"
        "graph_vectorstores/langchain_community.graph_vectorstores.extractors.html_link_extractor.HtmlLinkExtractor.html"
    )
    name = "HtmlLinkExtractor"
    icon = "LangChain"

    inputs = [
        StrInput(
            name="kind",
            display_name="Tipo de link",
            value="hyperlink",
            required=False,
            info="Tipo de conexão a ser extraída.",
        ),
        BoolInput(
            name="drop_fragments",
            display_name="Remover fragmentos",
            value=True,
            required=False,
            info="Eliminar fragmentos (# âncoras) das URLs extraídas.",
        ),
        DataInput(
            name="data_input",
            display_name="Entrada",
            info="Os textos dos quais extrair os links.",
            input_types=["Document", "Data"],
            required=True,
        ),
    ]

    def get_data_input(self) -> Any:
        return self.data_input

    def build_document_transformer(self) -> BaseDocumentTransformer:
        return LinkExtractorTransformer(
            [HtmlLinkExtractor(kind=self.kind, drop_fragments=self.drop_fragments)
             .as_document_extractor()]
        )
