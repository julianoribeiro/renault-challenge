import os
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from dotenv import load_dotenv

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

    def __post_init__(self):
        self.show_page()

    def show_page(self):
        csv_file = self.select_csv()
        if csv_file is not None:
            df = pd.read_csv(csv_file, sep=";")
            self.columns_layout(df, csv_file)
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


def main():
    EditCsvPage()


if __name__ == "__main__":
    main()
