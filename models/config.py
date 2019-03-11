
from utils.dictnamespace import DictNamespace
from configobj import ConfigObj as Config
from os.path import join

global_config = DictNamespace(Config(join('config', 'global.conf'), stringify=True))
