

class DictNamespace:
    '''
    This class can be used to create a namespace similar as types.SimpleNamespace where variables can be accessed via
    attribute indexes, but it reads the entries from an arbitrary dictionary.
    e.g:
    vars = { 'x': 1, 'y': 2, 'z' : 3 }
    V = DictNamespace(vars)

    print(V.x, V.y, V.z) -> 1, 2, 3

    Also indexation via [] operator can be used (like a regular dictionary) to fetch values

    print(V['x']) -> 1

    DictNamespace implements all metamethods as dict does. They just call the same metamethod on dict class passing as
    argument the dict instance specified on the constructor.
    This allows to do anything you can do with a dict:
    'x' in V
    iter(V)
    str(V), repr(V)
    len(V)

    If you specify a key in the dictionary passed on the constructor that starts and ends with '__'
    such entry value cannot be read with attribute indexation, but it can still be fetched using
    [] operator...

    V = DictNamespace( { '__len__' : 1 } )

    print(V['__len__']) -> 1


    If the dictionary passed to DictNamespace its modified, any change is reflected in the namespace and viceversa:

    d = { 'a' : 'Hello', 'b' : 'World!' }
    V = DictNamespace(d)
    print(V.a, V.b) -> Hello World!

    d['a'] = 'Bye'
    print(V.a,V.b) -> Bye World!

    V.a = 'Hello'
    V.b = 'There!'
    print(d) -> { 'a' : 'Hello', 'b' : 'There!' }


    Finally, by default if you access an entry on the dictionary via DictNamespace and its value is also a dictionary,
    such value is treated also as a DictNamespace object:

    V = DictNamespace({
        'a' : 1,
        'b' : 2,
        'foo' : {
            'c' : 3,
            'd' : 4
        }
    })

    print(V.foo.c, V.foo.d) -> 3, 4
    print(isinstance(V.foo, DictNamespace)) -> True

    You can change this behaviour setting the parameter recursive=False in the constructor (by default is True)

    V = DictNamespace({
        'a' : 1,
        'b' : 2,
        'foo' : {
            'c' : 3,
            'd' : 4
        }
    }, recursive = False)

    print(V.foo.c, V.foo.d) -> 3, 4
    print(isinstance(V.foo, DictNamespace)) -> False
    print(isinstance(V.foo, dict)) -> True
    '''

    def __init__(self, data, recursive = True):
        '''
        Initializes this object.
        :param data: Is the dictionary linked to this instance.
        :param recursive: Indicates if entries with dictionary values should be treated also like instances of DictNamespace or
        just regular dictionaries (by default is set to True)
        '''
        if not isinstance(data, dict):
            raise TypeError('Expected instance of type dict at argument 1, got {}'.format(type(data).__name__))
        if not isinstance(recursive, bool):
            raise TypeError('Expected instance of type bool at argument 2, got {}'.format(type(recursive).__name__))

        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'recursive', recursive)

    def __getattribute__(self, item):
        if type(item) != str:
            raise KeyError()
        if item.startswith('__') and item.endswith('__'):
            return super().__getattribute__(item)

        value = super().__getattribute__('data').__getitem__(item)
        if super().__getattribute__('recursive') and isinstance(value, dict):
            return DictNamespace(value, recursive=True)
        return value

    def __setattr__(self, item, value):
        if  type(item) != str:
            raise KeyError()
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError("'{}' object attribute '{}' is read-only".format(DictNamespace.__name__, item))
        return super().__getattribute__('data').__setitem__(item, value)


dictmetamethods = [
    '__repr__', '__hash__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__',
    '__ge__', '__iter__', '__len__', '__getitem__', '__setitem__', '__delitem__',
    '__contains__', '__doc__']

def proxy_method(cls, name):
    def proxy(self, *args, **kwargs):
        return getattr(cls, name)(object.__getattribute__(self, 'data'), *args, **kwargs)
    return proxy

for name in dictmetamethods:
    assert name in dict.__dict__
    if name.startswith('__') and name.endswith('__'):
        setattr(DictNamespace, name, proxy_method(dict, name))
