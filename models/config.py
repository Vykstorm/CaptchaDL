
from utils.dictnamespace import DictNamespace
from configobj import ConfigObj as Config
from validate import Validator as ConfigValidator
from os.path import join

global_config = Config(join('config', 'global.conf'),
                        configspec=join('config', 'global.spec.conf'),
                        stringify=True)

result = global_config.validate(ConfigValidator(), preserve_errors=True)
if result is not True:
    raise Exception('Invalid configuration: {}'.format(result))
global_config = DictNamespace(dict(global_config), recursive=True)

if __name__ == '__main__':
    print(global_config)
