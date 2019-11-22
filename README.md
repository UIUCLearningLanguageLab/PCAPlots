# PlotUtils

A collection of useful plotting functions previously used to analyze word embeddings of neural networks

## Figures

### Dendrogram-Heatmap

y axis labels for dendrogram heatmap:
in matplotlib, yticklabels by default start from top rather than origin (which is intuitive and labels rows of a matrix correctly)
but when using dendrograms, one needs to set the extent of the main axis, which then somehow reverses this behavior, such that
yticklabels start from the origin - this means they need to be reversed to label the rows of the matrix correctly


## TODO

* show example plots
* add example dummy data
* make plotting skip-gram neighbors work


## Tips & Tricks

### Fontsize

```python
plt.setp(ax.get_yticklabels(), fontsize=config.Fig.ax_fontsize)
plt.setp(ax.get_xticklabels(), fontsize=config.Fig.ax_fontsize)

### Tick Label Format

```python
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
```

### Embedding matplotlib figures in a html page (e.g. served by flask)

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
