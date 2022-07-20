import pandas as pd


def style_table(df: pd.DataFrame, axis: int, precision: int = 0):
    properties = {
        'font-size': '10pt',
        'background-color': 'white',
        'border-color': 'black',
        'border-style': 'solid',
        'border-width': '1px',
        'border-collapse': 'collapse',
        'width': '80px'
    }
    if precision == 0:
        str_format = "{:.0f}"
    elif precision == 1:
        str_format = "{:.1f}"
    elif precision == 2:
        str_format = "{:.2f}"
    elif precision == 3:
        str_format = "{:.3f}"
    elif precision == 4:
        str_format = "{:.4f}"
    else:
        raise ValueError

    return (df
            .style
            .set_properties(**properties)
            .background_gradient(cmap='OrRd', axis=axis)
            .format(str_format, subset=df.select_dtypes(include='number').columns)
            )
