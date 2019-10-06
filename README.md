# PlotUtils

A collection of useful plotting functions previously used to analyze word embeddings of neural networks

## TODO

* show example plots
* add example dummy data
* make plotting skip-gram neighbors work


## Embedding matplotlib figures in a html page (e.g. served by flask)

Use the following function to convert matplotlib figure objects to a format readadble by the browser:

```Python
def figs_to_imgs(*figs):
    imgs = []
    for fig in figs:
        print('Encoding fig...')
        figfile = BytesIO()
        fig.savefig(figfile, format='png')
        figfile.seek(0)
        img = base64.encodebytes(figfile.getvalue()).decode()
        imgs.append(img)
    return imgs
```

Next, install the `flask` which ships witht eh `Jinja2` templating language: 

```bash
pip install flask
```

And put the following in your html template file:

```html
<img src="data:image/png;base64,{{ img }}" class="custom-class" alt="Placeholder">
```
