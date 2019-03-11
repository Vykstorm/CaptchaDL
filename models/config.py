
from utils.dictnamespace import DictNamespace
from configobj import ConfigObj as Config


global_config = DictNamespace(Config('global.conf', stringify=True))
