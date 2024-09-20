import os
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from dotenv import load_dotenv
import plotly.express as px


load_dotenv()

st.set_page_config(layout="wide")


def get_csv_root(file_name):
    file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(file_path)
    grandparent_directory = os.path.dirname(parent_directory)
    return os.path.join(os.path.dirname(grandparent_directory), file_name)


CSV_DICT = {
    "costs": {
        "file": get_csv_root(os.getenv("COST_CSV")),
        "emoji": ":money_with_wings:†",
    },
    "materials": {
        "file": get_csv_root(os.getenv("MATERIAL_CSV")),
        "emoji": ":package:",
    },
    "suppliers": {
        "file": get_csv_root(os.getenv("SUPPLIERS_CSV")),
        "emoji": ":factory:",
    },
    "vehicles": {"file": get_csv_root(os.getenv("VEHICLES_CSV")), "emoji": ":truck:"},
}


@dataclass
class EditCsvPage:
    edited_df: pd.DataFrame = field(init=False)
    current_csv: str = field(init=False)

    def __post_init__(self):
        self.show_page()

    def show_page(self):
        csv_file = self.select_csv()
        if csv_file is not None:
            self.current_csv = csv_file
            df = pd.read_csv(csv_file, sep=";")
            self.columns_layout(df, csv_file)
            self.graphs()
        else:
            st.write("Please select a CSV file to edit.")

    def select_csv(self):
        cols = st.columns(len(CSV_DICT))
        for idx, (key, value) in enumerate(CSV_DICT.items()):
            emoji = value.get("emoji", "")
            if cols[idx].button(f"{emoji} {key}"):
                st.session_state["selected_csv"] = value["file"]
        return st.session_state.get("selected_csv", None)

    def columns_layout(self, df, csv_file):
        col1, col2 = st.columns([8, 2])
        with col1:
            self.show_table(df)
        with col2:
            self.save_csv(csv_file)

    def show_table(self, df):
        st.subheader("Editar Dados")
        edited_df = st.data_editor(df, use_container_width=True, key="data_editor")
        st.session_state["edited_df"] = edited_df

    def save_csv(self, csv_file):
        if st.button("Salvar Alterações"):
            edited_df = st.session_state.get("edited_df", None)
            if edited_df is not None:
                edited_df.to_csv(csv_file, sep=";", index=False)
                st.success("Arquivo salvo com sucesso!")
            else:
                st.error("No changes to save.")

    def graphs(self):
        current_file_name = os.path.basename(self.current_csv).split(".")[0]
        df = pd.read_csv(self.current_csv, sep=";")
        if current_file_name == "costs":
            self.costs_graphs(df)

        if current_file_name == "materials":
            self.material_graph(df)

        if current_file_name == "suppliers":
            self.suppliers_graph(df)

        if current_file_name == "vehicles":
            self.vehicles_graph(df)

    def costs_graphs(self, df):
        col1, col2 = st.columns(2)
        with col1:
            cols_to_plot = ["Km", "Van", "Leve", "Toco", "Truck", "Bitrem", "Fiorino"]
            df_plot = df[cols_to_plot]
            fig = px.line(
                df_plot,
                x="Km",
                y=cols_to_plot[1:],
                labels={"value": "Custo", "Km": "Quilometragem (Km)"},
                title="Variação de Custo por Quilometragem para Diferentes Veículos",
            )
            fig.update_layout(
                xaxis_title="Quilometragem (Km)",
                yaxis_title="Custo",
                legend_title_text="Veículos",
                hovermode="x unified",
            )

            st.plotly_chart(fig)

        with col2:
            km_selecionado = st.selectbox(
                "Selecione a Quilometragem", options=df["Km"].unique()
            )
            dados_filtrados = df[df["Km"] == km_selecionado]
            dados_filtrados = dados_filtrados.drop("Km", axis=1)
            dados_para_grafico = dados_filtrados.melt(
                var_name="Veículo", value_name="Custo"
            )
            fig = px.bar(
                dados_para_grafico,
                x="Veículo",
                y="Custo",
                title=f"Comparação de Custos para Quilometragem {km_selecionado}",
                labels={"Custo": "Custo", "Veículo": "Tipo de Veículo"},
            )
            fig.update_layout(
                xaxis_title="Tipo de Veículo", yaxis_title="Custo", xaxis_tickangle=-45
            )
            st.plotly_chart(fig)

    def material_graph(self, df):
        unit_selected = st.selectbox(
            "Selecione a unidade", options=["Quantidade_m3", "Quantidade_ton"]
        )
        fig = px.bar(
            df,
            x="Material",
            y=unit_selected,
        )
        st.plotly_chart(fig)

    def suppliers_graph(self, df):
        distance_data_numeric = df[
            ["Fornecedor1", "Fornecedor2", "Fornecedor3", "Fornecedor4", "Fornecedor5"]
        ].apply(pd.to_numeric)

        fig = px.imshow(
            distance_data_numeric.values,
            labels=dict(x="Para", y="De", color="Distância"),
            x=[
                "Fornecedor1",
                "Fornecedor2",
                "Fornecedor3",
                "Fornecedor4",
                "Fornecedor5",
            ],
            y=df["Fornecedor"],
            color_continuous_scale="Viridis",
            text_auto=True,
        )

        fig.update_layout(
            title="Mapa de Calor das Distâncias entre Fornecedores",
            xaxis_title="Para",
            yaxis_title="De",
        )

        st.plotly_chart(fig)

    def vehicles_graph(self, df):
        fig = px.bar(
            df,
            y="Quantidade",
            x="Veiculo",
        )
        st.plotly_chart(fig)


def main():
    EditCsvPage()


if __name__ == "__main__":
    main()
