import pandas as pd


def configure_pandas_display():
    pd.set_option('display.max_rows', 10000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    return


if "__main__" == __name__:
    configure_pandas_display()
