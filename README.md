<div align="center">
 <img src="images/logo.png" width="200"> 
</div>

A collection of useful plotting functions previously used to analyze word embeddings of neural networks

## Figures

### Dendrogram-Heatmap

y axis labels for dendrogram heatmap:
in matplotlib, yticklabels by default start from top rather than origin (which is intuitive and labels rows of a matrix correctly)
but when using dendrograms, one needs to set the extent of the main axis, which then somehow reverses this behavior, such that
yticklabels start from the origin - this means they need to be reversed to label the rows of the matrix correctly


### PCA across Time

Shows evolution of pattern of activations on 2 principal components across training time.

### Balanced Accuracy by Category

Shows a measure of categorization performance (e.g. balanced accuracy) for each word, in each category.

![balanced_accuracy_by_cat.png](images/balanced_accuracy_by_cat.png)

## Tips & Tricks

### Fontsize

```python
import matplotlib.pyplot as plt

fix, ax = plt.subplots()

plt.setp(ax.get_yticklabels(), fontsize=12)
plt.setp(ax.get_xticklabels(), fontsize=12)
```

### Tick Label Format

```python
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

fix, ax = plt.subplots()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
```

### Embedding matplotlib figures in a html page (e.g. served by flask)

Use the following function to convert matplotlib figure objects to a format readadble by the browser:

```Python
from io import BytesIO
import base64

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

## Compatibility

Developed using Python 3.8 on Ubuntu 16.04