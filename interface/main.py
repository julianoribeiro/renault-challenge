import os
import pandas as pd
import streamlit as st


def get_csv_root(file_name):
    file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(file_path)
    return os.path.join(os.path.dirname(parent_directory), file_name)


def get_materials_by_supplier(address_df, suppliers_df):
    return_dict = {}
    suppliers = address_df["Fornecedor"]
    for supplier in suppliers:
        supplier_row = suppliers_df[suppliers_df["Fornecedor"] == supplier].iloc[0]
        supplier_materials_list = supplier_row[supplier_row == "X"].index.tolist()
        return_dict[supplier] = supplier_materials_list

    return return_dict


def get_volume_by_supplier(material_df, address_df, suppliers_df):
    total_volume_dict = {}
    suppliers_materials_dict = get_materials_by_supplier(address_df, suppliers_df)
    for supplier, list_of_materials in suppliers_materials_dict.items():
        volume_list = [
            float(material_df[material_df["Material"] == x]["Quantidade_ton"])
            for x in list_of_materials
        ]
        total_volume = sum(volume_list)
        total_volume_dict[supplier] = total_volume

    return total_volume_dict


def main():
    address_df = pd.read_csv("interface/address_suppliers.csv", sep=";")
    suppliers_df = pd.read_csv(get_csv_root("suppliers.csv"), sep=";")
    materials_df = pd.read_csv(get_csv_root("materials.csv"), sep=";")
    volumes_dict = get_volume_by_supplier(materials_df, address_df, suppliers_df)
    address_df["volume"] = address_df["Fornecedor"].map(volumes_dict)
    st.map(address_df, longitude="longitude", latitude="latitude", size="volume")


if __name__ == "__main__":
    main()
