def modify_jupyter_notebook_html(
    input_html: str,
    output_html: str = None,
    font_size: int = 24,
    font_family: str = "Arial, sans-serif",
    max_width: str = "800px",
):
    """
    Modify the style of a Jupyter notebook HTML export.

    Parameters
    ----------
    input_html : str
        Path to the input HTML file.
    output_html : str, optional
        Path to the output HTML file. If not provided, the input HTML file will be modified in-place.
    font_size : int, optional
        The desired font size in pixels. Default is 16.
    font_family : str, optional
        The desired font family. Default is "Arial, sans-serif".
    max_width : str, optional
        The maximum width of the content. Default is "700px".

    """

    # Define the CSS to insert.
    css = f"""
    body {{
      font-size: {font_size}px;
      font-family: {font_family};
      margin: 0 auto;
      max-width: {max_width};
      text-align: left;
    }}
    div.output_text {{
      font-size: {font_size}px;
      font-family: {font_family};
      text-align: left;
    }}
    """

    # Read the HTML file.
    with open(input_html, "r", encoding="utf-8") as file:
        html = file.read()

    # Insert the CSS into the HTML.
    html = html.replace("</head>", f'<style type="text/css">{css}</style></head>')

    # Write the updated HTML back to the file.
    output_html = output_html if output_html else input_html
    with open(output_html, "w", encoding="utf-8") as file:
        file.write(html)