################ Schemish - based on Lispy: Scheme Interpreter in Python
## A modified version of Peter Norvig's Lispy (c) 2010-16; See http://norvig.com/lispy.html
## tailored to run pretty much only the code from The Little Schemer
##
## This version uses generators to avoid Python's recursion limit (aka trampolining).
## It can therefore handle deeply recursive Scheme programs like (! 100) but will still be really
## slow if you do things like redefine + and *, which are commented out in ls.scm but defined in
## the book. It also handles comments, has a nicer repl and can load files (see below). Tokenization
## is more robust, and it supports multiple expressions in a body. I have left everything else as is,
## and have not changed names of variables or functions unless necessary.
##
## The book code can be found at https://github.com/broop/little-schemer/
## Note: I've left most of the original comments, except where they became irrelevant or are no
## longer correct due to modification.
##
## Usage:
##   python3 schemish2.py ls.scm
##   >>> (! 10)
##   3628800

import operator as op
import re
import sys
import types
from collections import deque

################ Types

Symbol = str
List = list
Number = (int, float)

################ Parsing: parse, tokenize, and read_from_tokens

TOKEN_RE = re.compile(r"""
    (?P<WS>\s+) |
    (?P<COMMENT>;[^\n]*) |
    (?P<STRING>"[^"]*") |
    (?P<BOOL>\#[tf]) |
    (?P<PAREN>[()]) |
    (?P<QUOTE>') |
    (?P<ATOM>[^\s()';"]+)
""", re.VERBOSE)


def tokenize(s):
    return [m.group() for m in TOKEN_RE.finditer(s) if m.lastgroup not in ('WS', 'COMMENT')]


def parse(program):
    tokens = deque(tokenize(program))
    while tokens:
        yield read_from_tokens(tokens)


def read_from_tokens(tokens):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    token = tokens.popleft()
    match token:
        case '(':
            L = []
            while tokens[0] != ')':
                L.append(read_from_tokens(tokens))
            tokens.popleft()
            return L
        case ')':
            raise SyntaxError('unexpected )')
        case "'":  # support the ' shorthand for quoting used in the book
            return ['quote', read_from_tokens(tokens)]
        case _:
            return atom(token)


def atom(token):
    """Numbers become numbers; #t/#f become bools; else symbol."""
    match token:
        case '#t':
            return True
        case '#f':
            return False
        case s if s.startswith('"') and s.endswith('"'):
            return ('string', s[1:-1])
        case _:
            try:
                return int(token)
            except ValueError:
                try:
                    return float(token)
                except ValueError:
                    return Symbol(token)


################ Environments

def standard_env():
    """An environment with Scheme standard procedures needed for ls.scm."""
    env = Env()
    env.update({
        '+': op.add,
        '-': op.sub,
        '*': op.mul,
        '/': op.truediv,
        '=': op.eq,
        'modulo': op.mod,
        'car': lambda x: x[0],
        'cdr': lambda x: x[1:],
        'cons': lambda x, y: [x] + (y if isinstance(y, list) else [y]),
        'null?': lambda x: x == [],
        'pair?': lambda x: isinstance(x, list) and x != [],
        'number?': lambda x: isinstance(x, Number),
        'zero?': lambda x: x == 0,
        'even?': lambda x: x % 2 == 0,
        'add1': lambda x: x + 1,
        'sub1': lambda x: x - 1,
        'eq?': lambda x, y: x is y if isinstance(x, list) else x == y,
        'not': op.not_,
        # added for debugging and because they exist in ls.scm
        'display': lambda x: print(lispstr(x)),
        'printf': lambda fmt, *args: print(fmt.replace('~a', '{}').replace('~n', '\n')
                                           .format(*[lispstr(a) for a in args]))
    })
    return env


class Env(dict):
    """An environment: a dict of {'var': val} pairs, with an outer Env."""

    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


global_env = standard_env()


# nicer output
def lispstr(exp):
    """Convert a Python object back into a Lisp-readable string. Added a couple of things... """
    match exp:
        case True:
            return '#t'
        case False:
            return '#f'
        case list():
            return '(' + ' '.join(map(lispstr, exp)) + ')'
        case Procedure():
            return f'#<procedure {exp.parms}>'  #
        case _:
            return str(exp)


################ Procedures

class Procedure:
    """A user-defined Scheme procedure."""

    def __init__(self, parms, body_exprs, env):
        self.parms = parms
        self.body_exprs = body_exprs
        self.env = env

    def apply(self, args):
        """Evaluate body expressions in sequence, returning the last result."""
        env = Env(self.parms, args, self.env)
        result = None
        for exp in self.body_exprs:
            result = yield (exp, env)
        return result


# Generator-based eval
def _eval(x, env):
    """
    Evaluate an expression in an environment.


    - yield (expr, env) when it needs to evaluate a sub-expression
    - Receives the result back via send()
    - Returns the final value
    """
    match x:
        case ('string', s):
            return s
        case str(sym):  # variable reference
            return env.find(sym)[sym]

        # Constant literal (number, bool)
        case _ if not isinstance(x, list):
            return x

        case ['quote', val]:
            return val

        case ['cond', *clauses]:
            for clause in clauses:
                match clause:
                    case ['else', body]:
                        return (yield (body, env))
                    case [test, body]:
                        if (yield (test, env)):
                            return (yield (body, env))
            return None

        case ['define', var, exp]:
            val = yield (exp, env)
            env[var] = val
            return None

        case ['lambda', parms, *body_exprs]:
            return Procedure(parms, list(body_exprs), env)

        case ['and', *exprs]:
            for exp in exprs:
                val = yield (exp, env)
                if not val:
                    return False
            return True

        case ['or', *exprs]:
            for exp in exprs:
                val = yield (exp, env)
                if val:
                    return True
            return False

        # Procedure call
        case [op, *args_exprs]:
            proc = yield (op, env)
            args = []
            for exp in args_exprs:
                args.append((yield (exp, env)))
            if isinstance(proc, Procedure):
                return (yield from proc.apply(args))
            else:
                return proc(*args)


def eval(x, env=global_env):
    generator = _eval(x, env)
    if not isinstance(generator, types.GeneratorType):
        return generator

    stack = [generator]
    result = None

    while stack:
        try:
            expr, env = stack[-1].send(result)
            stack.append(_eval(expr, env))
            result = None

        except StopIteration as e:
            stack.pop()
            result = e.value  # good for now

    return result


def run_file(filename):
    with open(filename, 'r') as f:
        program = f.read()
    expressions = parse(program)
    for expr in expressions:
        eval(expr, global_env)


def repl():
    """A prompt-read-eval-print loop. Supports multi-line input"""
    while True:
        try:
            line = input('>>> ')
            if not line.strip():
                continue
            while line.count('(') > line.count(')'):
                line += ' ' + input('  ')
            result = eval(next(parse(line)))
            if result is not None:
                print(lispstr(result))
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    repl()
