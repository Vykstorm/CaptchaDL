
'''
This script shows an easy way to implement singleton pattern using decorators python feature.
Extracted from: https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
Original idea by Chih-Chung Chang modified by Vykstorm
'''

from inspect import isclass

class _Singleton:
    '''
    This is a class decorator (to decorate classes also) to implement the singleton pattern.
    Dont use this decorator directly, instead you can use the singleton method as decorator, see below.
    '''
    def __init__(self, cls, *args, **kwargs):
        '''
        Initializes this instance.
        :param cls: It must be the class to be decorated
        :param args: Additional position arguments to be included at singleton initialization
        :param kwargs: Additional keyword arguments to be included at singleton initialization
        '''
        if not isclass(cls):
            raise TypeError('Expected class at argument 1, got {}'.format(type(cls).__name__))

        self.cls = cls
        self.instance = None
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        '''
        This is called each time a new object of the decorated class need to be instantiated.
        It creates the singleton object if it doesnt exist yet and return it passing the args and kwargs indicated at
        this instance initialization as arguments to the singleton constructor.
        '''
        if self.instance is None:
            self.instance = self.cls(*self.args, **self.kwargs)
        return self.instance

def singleton(*args, **kwargs):
    '''
    This is a method that decorates class objects to implement singleton pattern.
    The next syntaxes can be used to mark a class as a singleton using this decorator:
    @singleton
    class Foo:
        ...

    @singleton(args = [...], kwargs = {...})
    class Foo:
        ...

    In the first example, singleton object constructor will not take any arguments.
    On the other, args (which must be an iterable object) places positional arguments in the singleton object
    constructor.
    Also kwargs (dictionary) entries will be sent as keyword arguments on the constructor.

    e.g:
    @singleton(args = (1,2,3), kwargs = {'x':4,'y':5})
    class Foo:
        def __init__(a,b,c, x, y):
            print(a+b+c,  x+y)

    Foo() will print "6, 9"

    :param args:
    :param kwargs:
    :return:
    '''
    if len(args) == 1 and len(kwargs) == 0:
        cls = args[0]
        return _Singleton(cls)

    if len(args) > 0:
        raise ValueError('Invalid decorator syntax. It must be: @singleton, @singleton() or @singleton([args = (...)], [kwargs = {...}])')

    if len(kwargs) == 0 or any([kwarg not in ('args', 'kwargs') for kwarg in kwargs]):
        invalid_args = [kwarg for kwarg in kwargs if kwarg not in ('args', 'kwargs')]
        raise TypeError('Unexpected decorator argument{}: "{}"'.format(
            's' if len(invalid_args) > 1 else '', ', '.join(invalid_args)))

    _args = tuple(kwargs['args']) if 'args' in kwargs else ()
    _kwargs = dict(kwargs['kwargs']) if 'kwargs' in kwargs else {}

    def _singleton(cls):
        return _Singleton(cls, *_args, **_kwargs)
    return _singleton


if __name__ == '__main__':
    '''
    This example illustrates the usage of singleton decorator
    '''
    @singleton(kwargs = {'limit': 900})
    class PerfectSquares:
        def __init__(self, limit=577):
            self.index = 1
            self.current = 1
            self.limit = limit

        def __iter__(self):
            return self

        def __next__(self):
            if self.current < self.limit:
                val = self.current
                self.index += 1
                self.current += (self.index << 1) - 1
                return val
            else:
                raise StopIteration()

    A = iter(PerfectSquares())
    B = iter(PerfectSquares())

    print('Perfect square numbers: ')
    try:
        while True:
            print('{:4d} {:4d} '.format(next(A), next(B)), end='')
    except StopIteration:
        pass
