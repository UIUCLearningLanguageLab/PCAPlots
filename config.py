class FigsConfigs(object):
    # global
    DPI = 96
    MAX_FIG_WIDTH = 7  # inches
    AXLABEL_FONT_SIZE = 14  # todo 12
    TICKLABEL_FONT_SIZE = 10
    LEG_FONTSIZE = 10
    LINEWIDTH = 2
    MARKERSIZE = 10
    FILL_ALPHA = 0.5

    # miscellaneous
    PCA_FREQ_THR = 100
    TAGS = ['UH', 'NN']
    PROBE_FREQ_YLIM = 1000
    NUM_PROBES_IN_QUARTILE = 5
    ALTERNATE_PROBES = ['cat', 'plate', 'dad', 'two', 'meat', 'tuesday', 'january']
    NUM_PCA_LOADINGS = 200
    CLUSTER_PCA_ITEM_ROWS = False
    CLUSTER_PCA_CAT_ROWS = True
    CLUSTER_PCA_CAT_COLS = True
    SKIPGRAM_MODEL_ID = 0
    NUM_PCS = 10  # how many principal_comps
    MAX_NUM_ACTS = 200  # max number of exemplar activations to use when working with exemplars
    LAST_PCS = [3]
    NUM_TIMEPOINTS_ACTS_CORR = 5
    DEFAULT_NUM_WALK_TIMEPOINTS = 5
    CAT_CLUSTER_XLIM = 1  # xlim for cat clusters (each fig should have same xlim enabling comparison)
    SVD_IS_TERMS = True
    NUM_EARLIEST_TOKENS = 100 * 1000
    NUM_PROBES_BAS_TIMEPOINTS = 6

    POS_TERMS_DICT = {'Interjection': ['okay', 'ow', 'whee', 'mkay', 'ohh', 'hey'],
                      'Onomatopoeia': ['moo', 'quack', 'meow', 'woof', 'vroom', 'oink'],
                      'Adverb': ['almost', 'afterwards', 'always', 'exactly', 'very', 'not'],
                      'Article': ['the', 'a', 'an'],
                      'Demonstrative': ['this', 'that', 'these', 'those'],
                      'Possessive': ['my', 'your', 'his', 'her', 'its', 'our'],
                      'Preposition': ['in', 'on', 'at', 'by', 'over', 'through', 'to'],
                      'Quantifier': ['few', 'any', 'much', 'many', 'most', 'some'],
                      'Conjunction': ['and', 'that', 'but', 'or', 'as', 'if'],
                      'Punctuation': ['.', '!', '?', ','],
                      'Number': ['two', 'three', 'seven', 'ten', 'hundred', 'thousand'],
                      'Pronoun': ['he', 'she', 'we', 'they', 'us', 'him']}
    PCA_ITEM_LIST1 = ['woof', 'oink', 'quack', 'meow', 'baa', 'mmm', 'whoa',
                      'zoom', 'hah', 'ohh', 'the', 'a', 'an', 'my', 'your',
                      'big', 'little', 'red', 'blue', 'good', 'couch', 'mirror',
                      'bathtub', 'table', 'rug', 'play', 'share', 'find', 'get', 'finish']
    PCA_ITEM_LIST2 = ['stir', 'chew', 'pour', 'drink', 'hold', 'juice', 'milk', 'cheerio', 'spaghetti',
                      'cracker', 'fork', 'spoon', 'napkin', 'knife', 'tissue',
                      'finger', 'hand', 'tongue', 'tooth', 'nose', 'grew', 'visit',
                      'ran', 'came', 'arrive', 'museum', 'aquarium', 'zoo', 'market',
                      'forest', 'hyena', 'flounder', 'beast', 'queen', 'prince', 'ago',
                      'merrily', 'bitsy', 'peep', 'weensie']

