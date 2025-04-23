from langflow.custom import Component
from langflow.io import DataFrameInput, MultilineInput, Output, StrInput
from langflow.schema.message import Message


class ParseDataFrameComponent(Component):
    display_name = "Analisar DataFrame"
    description = (
        "Converte um DataFrame em texto plano seguindo um modelo especificado. "
        "Cada coluna do DataFrame é tratada como uma possível chave de modelo, ex.: {coluna_nome}."
    )
    icon = "braces"
    name = "ParseDataFrame"
    legacy = True

    inputs = [
        DataFrameInput(
            name="df",
            display_name="DataFrame",
            info="O DataFrame a ser convertido em linhas de texto."
        ),
        MultilineInput(
            name="template",
            display_name="Template",
            info=(
                "O modelo para formatar cada linha. "
                "Use placeholders iguais aos nomes das colunas do DataFrame, por exemplo '{col1}', '{col2}'."
            ),
            value="{text}",
        ),
        StrInput(
            name="sep",
            display_name="Separador",
            advanced=True,
            value="\n",
            info="String que une todos os textos das linhas ao gerar a saída de texto única.",
        ),
    ]

    outputs = [
        Output(
            display_name="Texto",
            name="text",
            info="Todas as linhas combinadas em um único texto, cada linha formatada pelo modelo e separada por `sep`.",
            method="parse_data",
        ),
    ]

    def _clean_args(self):
        dataframe = self.df
        template = self.template or "{text}"
        sep = self.sep or "\n"
        return dataframe, template, sep

    def parse_data(self) -> Message:
        """Converte cada linha do DataFrame em uma string formatada pelo modelo

        e então as une com o separador `sep`. Retorna a string combinada.
        """
        dataframe, template, sep = self._clean_args()

        lines = []
        # Para cada linha do DataFrame, constrói um dicionário e formata
        for _, row in dataframe.iterrows():
            row_dict = row.to_dict()
            text_line = template.format(**row_dict)  # ex.: template="{text}", row_dict={"text": "Olá"}
            lines.append(text_line)

        # Une todas as linhas com o separador definido
        result_string = sep.join(lines)
        self.status = result_string  # armazena em self.status para logs de UI
        return Message(text=result_string)
