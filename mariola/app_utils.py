def save_pandas_df_info(df, filename):
    """
    Saves basic information and the last 3 rows of a pandas DataFrame to a file.

    Args:
        df (pd.DataFrame): A pandas DataFrame whose information will be saved.
        filename (str): The name of the file where the DataFrame information will be saved.

    Returns:
        None: The function writes the information to the specified file.

    Notes:
        The function saves the following information to the file:
        - The column names of the DataFrame.
        - The number of columns in the DataFrame.
        - The last 3 rows of the DataFrame without the index.
    """
    
    if df is None or df.empty or not filename:
        raise ValueError("df and filename must be provided and cannot be None.")
    
    with open(filename, 'w') as f:
        f.write("Pandas DataFrame:\n")
        f.write(str(df.columns) + "\n\n")
        f.write("Len Columns:\n")
        f.write(str(len(df.columns)) + "\n\n")
        f.write("Last 3 rows:\n")
        f.write(df.tail(3).to_string(index=False))