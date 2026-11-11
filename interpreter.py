"""Vexa language foundation, lexer, parser, and runtime — Chunks 1–14. Optional libraries load only through explicit use statements.

Language contract (EBNF; quoted words denote exact, case-sensitive tokens):
 program     = { statement }, EOF ;
 statement   = "use", ("gui" | "pandas" | "numpy" | "pytorch"), "."
             | "import", module_binding, { ",", module_binding }, "."
             | "from", module_path, "import", member_binding, { ",", member_binding }, "."
             | "save", table_atom, "as", "csv", "to", table_atom, "."
             | "let", IDENT, "be", expression, "."
             | "set", target, "to", expression, "."
             | "say", expression, "."
             | "print", expression, "."
             | "ask", expression, "into", IDENT, "."
             | "return", expression, "."
             | "if", expression, "then", block,
               { "else", "if", expression, "then", block }, [ "else", block ]
             | "repeat", expression, "times", block
             | "while", expression, block
             | "for", "each", IDENT, "in", expression, block
             | "for", IDENT, "from", expression, "to", expression,
               [ "step", expression ], block
             | "break", "." | "continue", "."
             | "define", IDENT, "with", "(", [ parameters ], ")", [ "->", return_type ], block
             | "class", IDENT, [ "(", (IDENT | "nn.Module"), ")" ], "{", { method }, "}"
             | expression, "." ;
 module_path = IDENT, { ".", IDENT } ; (* member dots use postfix adjacency *)
 module_binding = module_path, [ "as", IDENT ] ;
 member_binding = IDENT, [ "as", IDENT ] ;
 method      = "define", IDENT, "with", "(", [ parameters ], ")", [ "->", return_type ], block ;
 parameters  = IDENT, { ",", IDENT } ;
 target      = postfix ; (* must end in a name, index, or property *)
 block       = "{", { statement }, "}" ;
 expression  = disjunction ;
 disjunction = conjunction, { "or", conjunction } ;
 conjunction = negation, { "and", negation } ;
 negation    = "not", negation | comparison ;
 comparison  = sum, [ comparator, sum ] ;
 comparator  = "is", ( "equal", "to" | "not", "equal", "to"
             | ( "greater" | "less" ), "than", [ "or", "equal", "to" ] ) ;
 sum         = product, { ( "+" | "-" ), product } ;
 product     = unary, { ( "*" | "/" | "%" ), unary } ;
 unary       = ( "+" | "-" ), unary | power ;
 power       = postfix, [ "^", unary ] ;
 postfix     = primary, { "(", [ arguments ], ")" | "[", expression, "]" | ".", IDENT } ;
 arguments   = expression, { ",", expression } ;
 primary     = "super", "(", ")" | torch_phrase | array_phrase | table_phrase | NUMBER | STRING | IDENT | "true" | "false" | "null"
             | "(", expression, ")" | "[", [ arguments ], "]" ;

Table phrase grammar is specified in LANGUAGE_GUIDE section 18. Phrases match
contextually at primary-expression boundaries. Input slots are postfix operands
(literals, names, calls, lists, nested phrases, or parenthesized expressions).
first/last counts additionally accept unary signs and powers. A filter consumes
one longest-match comparator and one operand; ordinary enclosing operators remain
outside the phrase. No table vocabulary is added to KEYWORDS.

Lexical rules:
 ASCII identifiers: [A-Za-z_][A-Za-z0-9_]*. Keywords are reserved.
 NUMBER: digits optionally followed by '.' and one or more digits; no exponent
 notation. Thus 3.14. is NUMBER(3.14), DOT. A leading minus is an operator.
 Adjacent identifier suffixes and repeated fractional parts are lexical errors.
 Double-quoted strings allow \\n, \\r, \\t, \\" and \\\\ escapes only; raw
 newlines are forbidden. # starts a line comment outside strings. Whitespace
 is ignored. Every token records its 1-based start line and column.

Precedence, lowest first: or; and; not; comparison; + -; * / %; unary + -;
 ^; postfix call/index. Power is right associative (2^3^2 = 2^(3^2));
 -2^2 = -(2^2), and 2^-2 is valid. Comparisons do not chain. All other
 binary operators associate left. Comparison phrases consume the longest
 match, so 'is greater than or equal to' is one parser-level operator.
 Compound statements end at '}' without a period; simple statements require
 a period. Empty blocks/lists/argument lists are valid; trailing commas aren't.

Architecture and runtime contract (implemented incrementally):
 Lexer -> recursive-descent Parser with precedence layers -> immutable AST
 -> tree-walking Interpreter. Parser errors will report the offending token;
 runtime errors will report the AST token. No Python eval/exec is used.
 Values: float/int, str, bool, None (null), mutable lists, lexical closures,
 and built-in callables. Booleans are not numbers. Conditions and logical
 operands require bool; and/or short-circuit. Arithmetic requires numbers,
 except + also concatenates two strings or two lists. / yields float; % uses
 Python's remainder rule. ^ rejects complex results. Division by zero,
 invalid types, and non-finite numeric results are runtime errors.
 Equality is structural and type-sensitive except int/float compare numerically;
 ordering accepts two numbers or two strings. Lists use integer, nonnegative,
 in-range indices (bool excluded), with assignment on lists only. Arrays additionally support read indexing (section 19).
 No implicit coercion; say/print display strings unquoted, booleans/null lowercase,
 and lists recursively with quoted/escaped string elements.
 Each function call has a child environment of its captured definition scope.
 Classes accept only uniquely named methods; calling a class creates an instance
 with its own properties and invokes __init__ with the supplied arguments.
 Without __init__, classes accept zero arguments. Constructors must return null. Methods capture lexical scope and
 bind self implicitly in the call environment. Adjacent .IDENT is property access;
 a bare dot terminates a statement. Properties shadow methods. class Child(Parent) supports single inheritance;
 methods and constructors resolve child-first and preserve the child self.
 super().method(...) searches above the defining class, retaining the same self.
 Counter for loops evaluate integer bounds/step once, use an exclusive end,
 and create a local scope with an independent automatic counter; step defaults
 to 1 and must not be zero.
 Blocks otherwise share their enclosing scope. let, class, and define reject duplicate local
 bindings; set updates the nearest existing binding. ask reads a string and
 assigns an existing binding, or defines it locally if absent. for each binds
 its variable locally (updates it on later iterations), iterates a snapshot of
 a list and leaves the binding afterward. repeat requires a nonnegative int.
 repeat evaluates its count once; while reevaluates its boolean condition before
 each iteration. break/continue target the innermost loop and are invalid outside
 loops, even in unreachable branches. List iteration uses a shallow snapshot.
 Function definitions bind closures when executed, allowing recursion; arity
 is exact, duplicate parameters are invalid. return outside a function fails;
 falling off a function returns null. Calls evaluate callee then args left to
 right; assignments resolve their target before evaluating the right side.
 random(low, high) is a normal global built-in binding, inclusive integer
 bounds (bool excluded), requiring low <= high; it can be shadowed locally.
 Built-ins occupy a parent of the program environment. No forward hoisting.
"""

from __future__ import annotations

import math
import random as random_module
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KEYWORDS = frozenset("""
let be set to say print ask into return if then else repeat times while for each in from step
 use import class super define with true false null is equal not greater less than or and break continue
""".split())

# Public editor metadata. Runtime registration derives arity from these signatures.
BUILTIN_METADATA = {'random': [['low', 'high'],
            'Returns a uniformly chosen integer between inclusive integer bounds; low must not '
            'exceed high.'],
 'to_int': [['text'],
            'Parses a signed decimal string as an integer, trimming whitespace. Fractions and '
            'exponents are rejected.'],
 'to_float': [['text'],
              'Parses a finite float from a decimal string, optionally with exponent; trims '
              'whitespace.'],
 'to_string': [['value'], 'Returns Vexa’s display representation of any value.'],
 'length': [['value'], 'Counts list elements or Unicode code points in a string.'],
 'abs': [['number'], 'Returns the nonnegative magnitude, preserving integer/float kind.'],
 'sqrt': [['number'],
          'Returns a float square root. Input must be nonnegative and within host range.'],
 'floor': [['number'], 'Returns an integer rounded toward negative infinity.'],
 'ceil': [['number'], 'Returns an integer rounded toward positive infinity.'],
 'round': [['number'], 'Returns the nearest integer; exact halfway cases round to even.'],
 'min': [['a', 'b'], 'Returns the smaller of two numbers, first on a tie.'],
 'max': [['a', 'b'], 'Returns the larger of two numbers, first on a tie.'],
 'append': [['items', 'value'], 'Adds a value to the end of a list, mutating it. Returns null.'],
 'pop': [['items'], 'Removes and returns the last element of a nonempty list.']}

GUI_METADATA = {
    'make_window': [['title', 'width', 'height'], 'Requires use gui. Creates a desktop window; dimensions are positive integer pixels.'],
    'make_button': [['window', 'text', 'callback'], 'Requires use gui. Creates a button with a zero-parameter Vexa function callback.'],
    'make_label': [['window', 'text'], 'Requires use gui. Creates a text label.'],
    'make_text_input': [['window'], 'Requires use gui. Creates a single-line text input.'],
    'get_text': [['widget'], 'Requires use gui. Reads an input, label, button, or window title.'],
    'set_text': [['widget', 'text'], 'Requires use gui. Updates widget text or window title; returns null.'],
    'place': [['widget', 'x', 'y', 'width', 'height'], 'Requires use gui. Positions a child widget using integer pixels; returns null.'],
    'show_window': [['window'], 'Requires use gui. Processes events until this window closes; returns null.'],
    'close_window': [['window'], 'Requires use gui. Destroys a window and its widgets; returns null.'],
    'style': [['widget', 'background', 'foreground', 'font_size'], 'Requires use gui. Sets colors and positive integer font size; null leaves an option unchanged. Windows support background only.'],
}
BUILTIN_METADATA.update(GUI_METADATA)

COMPARISON_PHRASES = ('is equal to', 'is not equal to', 'is greater than', 'is less than', 'is greater than or equal to', 'is less than or equal to')

# Contextual phrases, not reserved lexer keywords. Shared with editor assets.
TABLE_PHRASES = {
    'use pandas': ['use pandas.', 'Enable real pandas table support for this interpreter.'],
    'read csv from': ['read csv from "${1:students.csv}"', 'Read a local UTF-8 CSV file with a header row.'],
    'first rows of': ['first ${1:5} rows of ${2:students}', 'Return a new table containing the first N rows.'],
    'last rows of': ['last ${1:3} rows of ${2:students}', 'Return a new table containing the last N rows.'],
    'number of rows in': ['number of rows in ${1:students}', 'Count table rows.'],
    'number of columns in': ['number of columns in ${1:students}', 'Count table columns.'],
    'columns of': ['columns of ${1:students}', 'Return column names as a Vexa list.'],
    'column from': ['column "${1:name}" from ${2:students}', 'Return a pandas column (Series) copy.'],
    'rows of': ['rows of ${1:students} where column "${2:score}" is greater than ${3:70}', 'Filter rows using one English comparison; missing cells never match.'],
    'drop missing rows from': ['drop missing rows from ${1:students}', 'Return a table without rows containing missing cells.'],
    'fill missing in column': ['fill missing in column "${1:score}" of ${2:students} with ${3:0}', 'Return a table with missing cells in this column replaced.'],
    'save': ['save ${1:students} as csv to "${2:output.csv}".', 'Write a local CSV without an index. Refuses existing files.'],
    'table with columns': ['table with columns ["${1:name}", "${2:score}"] and rows [["${3:Alex}", ${4:90}]]', 'Construct a table from unique string column names and equally sized rows.'],
}
for _stat in ('average', 'total', 'minimum', 'maximum', 'count'):
    TABLE_PHRASES[_stat + ' of column'] = [
        _stat + ' of column "${1:score}" in ${2:students}',
        ('Count nonmissing cells.' if _stat == 'count' else
         'Calculate ' + _stat + ' of numeric cells, skipping missing values; no text conversion.')]

# Numerical phrases share the existing contextual phrase/editor pipeline.
ARRAY_PHRASES = {
    'table from array': ['table from array ${1:matrix} with columns ${2:["x", "y"]}', 'Convert a two-dimensional array to a pandas table; may share storage.'],
    'use numpy': ['use numpy.', 'Enable optional real NumPy arrays.'],
    'array with': ['array with ${1:[1, 2, 3]}', 'Create an array from a rectangular list.'],
    'array from': ['array from ${1:scores}', 'Convert a pandas column/table or retain an existing array.'],
    'array range': ['array from ${1:1} to ${2:10}', 'Integer range with both endpoints included, step one.'],
    'array of zeros with shape': ['array of zeros with shape ${1:[3, 3]}', 'Float array filled with zeros.'],
    'array of ones with shape': ['array of ones with shape ${1:[2, 4]}', 'Float array filled with ones.'],
    'random array with shape': ['random array with shape ${1:[2, 3]}', 'Uniform random floats in [0, 1).'],
    'shape of': ['shape of ${1:matrix}', 'Dimension lengths as a Vexa list.'],
    'reshape': ['reshape ${1:matrix} to ${2:[4]}', 'Change shape without changing element count; may share storage.'],
    'transpose': ['transpose ${1:matrix}', 'Reverse dimension order; shares storage.'],
    'add': ['add ${1:5} to ${2:a}', 'Elementwise addition with NumPy broadcasting.'],
    'add arrays': ['add arrays ${1:a} and ${2:b}', 'Elementwise addition with NumPy broadcasting.'],
    'subtract': ['subtract ${1:5} from ${2:a}', 'Subtract the first operand from the second elementwise.'],
    'multiply': ['multiply ${1:a} by ${2:2}', 'Elementwise multiplication.'],
    'divide': ['divide ${1:a} by ${2:2}', 'Elementwise division; zero division is an error.'],
    'matrix multiply': ['matrix multiply ${1:a} by ${2:b}', 'NumPy matmul; inner dimensions must agree.'],
    'slice of': ['slice of ${1:a} from ${2:0} to ${3:2}', 'First-axis slice, start included and stop excluded; shares storage.'],
}
for _phrase, _meaning in {
    'average': 'Mean', 'total': 'Sum', 'minimum': 'Minimum', 'maximum': 'Maximum',
    'standard deviation': 'Population standard deviation (ddof=0)',
    'square root': 'Elementwise square root', 'absolute values': 'Elementwise absolute value',
    'sine': 'Elementwise sine in radians', 'cosine': 'Elementwise cosine in radians',
    'exponential': 'Elementwise exponential', 'logarithm': 'Elementwise natural logarithm',
}.items():
    ARRAY_PHRASES[_phrase + ' of'] = [_phrase + ' of ${1:a}', _meaning + ' of a numeric array; statistics use all elements.']

TABLE_PHRASES.update(ARRAY_PHRASES)

LAYERS = dict(zip(
    'conv1d conv2d conv3d linear relu sigmoid tanh flatten dropout batchnorm1d batchnorm2d batchnorm3d maxpool1d maxpool2d maxpool3d embedding sequential'.split(),
    'Conv1d Conv2d Conv3d Linear ReLU Sigmoid Tanh Flatten Dropout BatchNorm1d BatchNorm2d BatchNorm3d MaxPool1d MaxPool2d MaxPool3d Embedding Sequential'.split()))
ML_PHRASES = {
    'use pytorch': ['use pytorch.', 'Enable optional genuine PyTorch tensors and training.'],
    'tensor with': ['tensor with ${1:[1, 2, 3]}', 'Create a tensor; optional with options (dtype as "float32", device as "cpu", requires_grad as true).'],
    'columns from': ['columns ${1:["size", "bedrooms"]} from ${2:data}', 'Select pandas feature columns in the requested order. Requires use pandas and use pytorch.'],
    'tensor from': ['tensor from ${1:features}', 'Convert numeric lists, NumPy arrays or pandas data; rejects missing or text values.'],
    'predict with': ['predict with ${1:model} using ${2:inputs}', 'Forward pass preserving gradients for training.'],
    'infer with': ['infer with ${1:model} using ${2:inputs}', 'Forward pass without gradients; set evaluation mode separately.'],
    'mean squared error loss': ['mean squared error loss', 'Construct torch.nn.MSELoss.'],
    'cross entropy loss': ['cross entropy loss', 'Construct torch.nn.CrossEntropyLoss.'],
    'calculate loss using': ['calculate loss using ${1:criterion} with (${2:predictions}, ${3:targets})', 'Calculate loss; MSE requires equal shapes.'],
    'adam optimizer for': ['adam optimizer for ${1:model} with learning rate ${2:0.001}', 'Adam over model parameters.'],
    'sgd optimizer for': ['sgd optimizer for ${1:model} with learning rate ${2:0.01}', 'SGD over model parameters.'],
    'clear gradients of': ['clear gradients of ${1:optimizer}.', 'Clear accumulated gradients.'],
    'calculate gradients from': ['calculate gradients from ${1:loss}.', 'Backpropagate through the real PyTorch graph.'],
    'update weights using': ['update weights using ${1:optimizer}.', 'Apply an optimizer step.'],
    'training mode for': ['training mode for ${1:model}.', 'Enable training behavior such as dropout.'],
    'evaluation mode for': ['evaluation mode for ${1:model}.', 'Enable evaluation behavior; does not disable gradients.'],
    'best available device': ['best available device', 'Return visible torch.device: CUDA when available, otherwise CPU.'],
    'cuda available': ['cuda available', 'Check torch.cuda.is_available().'],
    'cuda device count': ['cuda device count', 'Number of visible CUDA devices.'],
    'cuda device name': ['cuda device name', 'Name of current CUDA device; fails when unavailable.'],
    'device of': ['device of ${1:numbers}', 'Inspect tensor device.'],
    'dtype of': ['dtype of ${1:numbers}', 'Inspect tensor data type.'],
    'parameters of': ['parameters of ${1:model}', 'List of genuine model parameters, preserving identity.'],
    'scalar value of': ['scalar value of ${1:loss}', 'Extract one scalar with item(); ends gradient tracking for this result.'],
    'tensor values of': ['tensor values of ${1:numbers}', 'Explicit detached CPU list conversion.'],
    'convert tensor': ['convert tensor ${1:numbers} to ${2:"float32"}', 'Change tensor dtype while retaining autograd.'],
    'move': ['move ${1:numbers} to ${2:device}', 'Move tensor or model; unavailable CUDA fails.'],
    'load images from': ['load images from ${1:"dataset/"}', 'ImageFolder with default ToTensor; optional with transforms TRANSFORM.'],
    'resize images to': ['resize images to ${1:[32, 32]}', 'Construct torchvision Resize.'],
    'images to tensors': ['images to tensors', 'Construct torchvision ToTensor.'],
    'normalize images with': ['normalize images with (${1:[0.5]}, ${2:[0.5]})', 'Construct Normalize(mean, std).'],
    'compose transforms with': ['compose transforms with (${1:resize}, ${2:convert})', 'Compose transforms in order.'],
    'make batches from': ['make batches from ${1:images} with size ${2:32}', 'Lazy DataLoader; optional with options (shuffle as true, num_workers as 0).'],
    'load mnist from': ['load mnist from ${1:"data"} with options (download as false)', 'MNIST; download defaults false, train defaults true.'],
    'load cifar10 from': ['load cifar10 from ${1:"data"} with options (download as false)', 'CIFAR10; download defaults false.'],
    'save model weights of': ['save model weights of ${1:model} to ${2:"model.pth"}.', 'Save state dictionary.'],
    'load model weights into': ['load model weights into ${1:model} from ${2:"model.pth"}.', 'Load weights_only=True into an existing compatible architecture.'],
    'save tensor': ['save tensor ${1:numbers} to ${2:"tensor.pth"}.', 'Save tensor data.'],
    'load tensor from': ['load tensor from ${1:"tensor.pth"}', 'Load tensor data on CPU using weights_only=True.'],
    'python module': ['python module ${1:"torch.nn"}', 'Explicit allowlisted advanced module access after use pytorch.'],
    'python member': ['python member ${1:"Linear"} of ${2:nn}', 'Access a public Python member; advanced interoperability.'],
    'python call': ['python call ${1:function} with (${2:arguments})', 'Call an advanced Python API; optional keyword arguments use name as value.'],
}
for _name in LAYERS:
    ML_PHRASES[_name + ' layer with'] = [_name + ' layer with (${1})', 'Construct genuine torch.nn.' + LAYERS[_name] + '; keywords use name as value.']
TABLE_PHRASES.update(ML_PHRASES)

def language_metadata() -> dict:
    """Return JSON-serializable language data without creating a runtime."""
    return {'schema_version': 1, 'keywords': sorted(KEYWORDS),
            'builtins': {name: [list(params), doc]
                         for name, (params, doc) in BUILTIN_METADATA.items()},
            'comparisons': list(COMPARISON_PHRASES), 'table_phrases': TABLE_PHRASES,
            'phrase_libraries': {phrase: ('pandas and numpy' if phrase == 'table from array' else
                                          'pytorch' if phrase in ML_PHRASES else 'numpy' if phrase in ARRAY_PHRASES else 'pandas')
                                 for phrase in TABLE_PHRASES}}

PUNCTUATION = {
    '.': 'DOT', ',': 'COMMA', '(': 'LPAREN', ')': 'RPAREN',
    '[': 'LBRACKET', ']': 'RBRACKET', '{': 'LBRACE', '}': 'RBRACE',
    '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH',
    '%': 'PERCENT', '^': 'CARET',
}


@dataclass(frozen=True)
class Token:
    kind: str
    lexeme: str
    value: Any
    line: int
    column: int
    source_path: Path | None = None


class VexaError(Exception):
    """Base diagnostic with a source location."""

    def __init__(self, message: str, line: int, column: int, source_path: Path | None = None):
        self.line, self.column = line, column
        self.source_path = source_path
        super().__init__(f'{line}:{column}: {message}')


class LexError(VexaError):
    pass


class ParseError(VexaError):
    pass


class VexaRuntimeError(VexaError):
    pass


RETURN_TYPES = {'int': int, 'float': float, 'string': str, 'bool': bool,
                'null': type(None), 'list': list, 'tensor': None}


class Lexer:
    """Single-pass scanner. Comparison phrases remain separate keyword tokens."""

    def __init__(self, source: str, source_path: Path | None = None):
        self.source = source
        self.source_path = source_path
        self.offset = 0
        self.line = 1
        self.column = 1

    def _peek(self) -> str:
        return self.source[self.offset:self.offset + 1]

    def _advance(self) -> str:
        char = self._peek()
        self.offset += 1
        if char == '\r':
            self.line += 1
            self.column = 1
        elif char == '\n':
            # CRLF is one newline; a standalone CR also starts a new line.
            if self.offset < 2 or self.source[self.offset - 2] != '\r':
                self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def tokenize(self) -> list[Token]:
        tokens = []
        while self._peek():
            char = self._peek()
            if char in ' \t\r\n':
                self._advance()
                continue
            if char == '#':
                while self._peek() and self._peek() not in '\r\n':
                    self._advance()
                continue
            start, line, column = self.offset, self.line, self.column
            value: Any = None
            if char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_':
                self._advance()
                while self._peek() and self._peek() in (
                    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
                ):
                    self._advance()
                word = self.source[start:self.offset]
                kind = word.upper() if word in KEYWORDS else 'IDENT'
                value = {'true': True, 'false': False, 'null': None}.get(word, word)
                if word == 'null':
                    value = None
            elif char in '0123456789':
                while self._peek() and self._peek() in '0123456789':
                    self._advance()
                if (self._peek() == '.' and
                        self.offset + 1 < len(self.source)
                        and self.source[self.offset + 1] in '0123456789'):
                    self._advance()
                    while self._peek() and self._peek() in '0123456789':
                        self._advance()
                raw = self.source[start:self.offset]
                following = self._peek()
                if following and following in (
                    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
                ):
                    raise LexError(
                        'Invalid numeric literal: identifier suffixes and exponent notation '
                        'are not supported', line, column)
                if ('.' in raw and following == '.' and
                        self.offset + 1 < len(self.source) and
                        self.source[self.offset + 1] in '0123456789'):
                    raise LexError('Invalid numeric literal: multiple decimal points',
                                   line, column)
                kind = 'NUMBER'
                try:
                    value = float(raw) if '.' in raw else int(raw)
                except ValueError:
                    raise LexError('Invalid numeric literal', line, column) from None
                if isinstance(value, float) and not math.isfinite(value):
                    raise LexError('Numeric literal is not finite', line, column)
            elif char == '"':
                self._advance()
                decoded = []
                escapes = {'n': '\n', 'r': '\r', 't': '\t', '"': '"', '\\': '\\'}
                while self._peek() != '"':
                    if not self._peek() or self._peek() in '\r\n':
                        raise LexError('Unterminated string', line, column)
                    current = self._advance()
                    if current == '\\':
                        escape = self._peek()
                        if not escape or escape in '\r\n':
                            raise LexError('Unterminated string', line, column)
                        if escape not in escapes:
                            raise LexError(f'Invalid string escape \\{escape}',
                                           self.line, self.column)
                        self._advance()
                        current = escapes[escape]
                    decoded.append(current)
                self._advance()
                kind, value = 'STRING', ''.join(decoded)
            elif char == '-' and self.source[self.offset:self.offset + 2] == '->':
                self._advance()
                self._advance()
                kind = 'ARROW'
            elif char in PUNCTUATION:
                kind = PUNCTUATION[self._advance()]
            else:
                raise LexError(f'Unexpected character {char!r}', line, column)
            tokens.append(Token(kind, self.source[start:self.offset], value, line, column,
                                self.source_path))
        tokens.append(Token('EOF', '', None, self.line, self.column, self.source_path))
        return tokens


# All AST nodes carry the first relevant token for diagnostics.
@dataclass(frozen=True)
class Node:
    token: Token


@dataclass(frozen=True)
class Expr(Node):
    pass


@dataclass(frozen=True)
class Literal(Expr):
    value: Any


@dataclass(frozen=True)
class Super(Expr):
    pass


@dataclass(frozen=True)
class Name(Expr):
    identifier: str


@dataclass(frozen=True)
class Unary(Expr):
    operator: str
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    operator: str
    right: Expr


@dataclass(frozen=True)
class Call(Expr):
    callee: Expr
    arguments: tuple[Expr, ...]


@dataclass(frozen=True)
class Property(Expr):
    object: Expr
    name: str


@dataclass(frozen=True)
class Index(Expr):
    collection: Expr
    index: Expr


@dataclass(frozen=True)
class ListLiteral(Expr):
    elements: tuple[Expr, ...]


@dataclass(frozen=True)
class TableExpr(Expr):
    operation: str
    arguments: tuple[Expr, ...]
    comparison: str = ''


@dataclass(frozen=True)
class ArrayExpr(Expr):
    operation: str
    arguments: tuple[Expr, ...]


@dataclass(frozen=True)
class TorchExpr(Expr):
    operation: str
    arguments: tuple[Expr, ...] = ()
    keywords: tuple[tuple[str, Expr], ...] = ()


@dataclass(frozen=True)
class Stmt(Node):
    pass


@dataclass(frozen=True)
class Import(Stmt):
    bindings: tuple[tuple[tuple[str, ...], str], ...]
    module: tuple[str, ...] | None = None


@dataclass(frozen=True)
class UseLibrary(Stmt):
    name: str


@dataclass(frozen=True)
class SaveCsv(Stmt):
    table: Expr
    path: Expr


@dataclass(frozen=True)
class Block(Node):
    statements: tuple[Stmt, ...]


@dataclass(frozen=True)
class UseGui(Stmt):
    pass


@dataclass(frozen=True)
class Let(Stmt):
    name: str
    value: Expr


@dataclass(frozen=True)
class Assign(Stmt):
    target: Name | Index | Property
    value: Expr


@dataclass(frozen=True)
class Say(Stmt):
    value: Expr


@dataclass(frozen=True)
class Print(Stmt):
    value: Expr


@dataclass(frozen=True)
class Ask(Stmt):
    prompt: Expr
    name: str


@dataclass(frozen=True)
class Return(Stmt):
    value: Expr


@dataclass(frozen=True)
class Break(Stmt):
    pass


@dataclass(frozen=True)
class Continue(Stmt):
    pass


@dataclass(frozen=True)
class ExpressionStatement(Stmt):
    expression: Expr


@dataclass(frozen=True)
class If(Stmt):
    branches: tuple[tuple[Expr, Block], ...]
    otherwise: Block | None


@dataclass(frozen=True)
class Repeat(Stmt):
    count: Expr
    body: Block


@dataclass(frozen=True)
class While(Stmt):
    condition: Expr
    body: Block


@dataclass(frozen=True)
class ForEach(Stmt):
    name: str
    iterable: Expr
    body: Block


@dataclass(frozen=True)
class CounterFor(Stmt):
    name: str
    start: Expr
    end: Expr
    step: Expr | None
    body: Block


@dataclass(frozen=True)
class Define(Stmt):
    name: str
    parameters: tuple[str, ...]
    body: Block
    return_type: str | None = None


@dataclass(frozen=True)
class ClassDeclaration(Stmt):
    name: str
    methods: tuple[Define, ...]
    parent: str | None = None


@dataclass(frozen=True)
class Program(Node):
    statements: tuple[Stmt, ...]


class Environment:
    """Lexical bindings; lookup/assignment walk parents, definitions stay local."""

    def __init__(self, parent: Environment | None = None):
        self.parent = parent
        self.values: dict[str, Any] = {}
        self.method_context: tuple[ClassValue | None, Instance | None] = (None, None)

    def define(self, name: str, value: Any, token: Token) -> None:
        if name in self.values:
            raise VexaRuntimeError(f'Duplicate binding {name!r}', token.line, token.column,
                                   token.source_path)
        self.values[name] = value

    def _owner(self, name: str, token: Token) -> Environment:
        scope: Environment | None = self
        while scope is not None:
            if name in scope.values:
                return scope
            scope = scope.parent
        raise VexaRuntimeError(f'Undefined name {name!r}', token.line, token.column,
                               token.source_path)

    def get(self, name: str, token: Token) -> Any:
        return self._owner(name, token).values[name]

    def assign(self, name: str, value: Any, token: Token) -> None:
        self._owner(name, token).values[name] = value


@dataclass(frozen=True, eq=False)
class Module:
    path: Path
    environment: Environment


@dataclass(frozen=True, eq=False)
class Function:
    declaration: Define
    closure: Environment
    defining_class: ClassValue | None = None


@dataclass(frozen=True, eq=False)
class ClassValue:
    name: str
    methods: dict[str, Function]
    parent: ClassValue | None = None

    @property
    def torch_base(self):
        return self.name == 'nn.Module' or (self.parent is not None and self.parent.torch_base)

    def find_method(self, name: str) -> Function | None:
        klass = self
        while klass is not None:
            if name in klass.methods:
                return klass.methods[name]
            klass = klass.parent
        return None


@dataclass(eq=False)
class Instance:
    klass: ClassValue
    properties: dict[str, Any]
    native: Any = None


@dataclass(frozen=True, eq=False)
class BoundMethod:
    function: Function
    instance: Instance


@dataclass(frozen=True, eq=False)
class SuperProxy:
    parent: ClassValue | None
    instance: Instance


@dataclass(frozen=True, eq=False)
class ModuleInitializer:
    instance: Instance


@dataclass(frozen=True, eq=False)
class Builtin:
    name: str
    arity: int


class Parser:
    """Recursive-descent parser; operators retain their source spelling in ASTs.

    Supports expressions, let/set/say/print, conditionals, loops, and functions. Input is a
    lexer token stream.
    """

    def __init__(self, tokens: list[Token]):
        if not tokens or tokens[-1].kind != 'EOF':
            raise ValueError('Parser requires a token stream ending in EOF')
        self.tokens = tokens
        self.position = 0
        self.loop_depth = 0
        self.function_depth = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.position]

    def _match(self, *kinds: str) -> Token | None:
        # New loop keywords remain usable as existing identifier names.
        if self.current.kind in kinds or ('IDENT' in kinds and
                                           self.current.kind in {'FROM', 'STEP'}):
            token = self.current
            if token.kind != 'EOF':
                self.position += 1
            return token
        return None

    def _error(self, message: str) -> ParseError:
        token = self.current
        found = repr(token.lexeme) if token.kind != 'EOF' else 'end of input'
        return ParseError(f'{message}; found {found}', token.line, token.column)

    def _expect(self, kind: str, description: str) -> Token:
        token = self._match(kind)
        if token is None:
            raise self._error(f'Expected {description}')
        return token

    def parse(self) -> Program:
        start = self.current
        statements = []
        try:
            while self.current.kind != 'EOF':
                statements.append(self._statement())
        except RecursionError:
            raise self._error('Program nesting exceeds parser recursion limit') from None
        return Program(start, tuple(statements))

    def _module_path(self) -> tuple[str, ...]:
        parts = [self._expect('IDENT', 'module name').value]
        while self._property_dot():
            self._expect('DOT', "'.'")
            parts.append(self._expect('IDENT', 'module name').value)
        return tuple(parts)

    def _import(self, token: Token) -> Import:
        module = None
        if token.kind == 'FROM':
            module = self._module_path()
            self._expect('IMPORT', "'import'")
        bindings = []
        while True:
            path = (self._expect('IDENT', 'imported name').value,) if module else self._module_path()
            name = path[-1]
            if self._words('as'):
                self._take_words('as')
                name = self._expect('IDENT', 'import alias').value
            bindings.append((path, name))
            if not self._match('COMMA'):
                break
        self._expect('DOT', "'.' after import")
        return Import(token, tuple(bindings), module)

    def _statement(self) -> Stmt:
        token = self.current
        if token.kind == 'IMPORT' or (token.kind == 'FROM' and
                                     self.tokens[self.position + 1].kind in {'IDENT', 'FROM', 'STEP'}):
            self._match(token.kind)
            return self._import(token)
        if self._match('CLASS'):
            name = self._expect('IDENT', 'class name')
            parent = None
            if self._match('LPAREN'):
                parent = self._expect('IDENT', 'parent class name').value
                while self._match('DOT'):
                    parent += '.' + self._expect('IDENT', 'allowlisted parent class member').value
                self._expect('RPAREN', "')' after single parent class name")
            self._expect('LBRACE', "'{' before methods")
            methods = []
            names = set()
            while self.current.kind != 'RBRACE':
                if self.current.kind != 'DEFINE':
                    raise self._error('Class bodies may only contain method definitions')
                method = self._statement()
                if method.name in names:
                    raise ParseError(f'Duplicate method {method.name!r}',
                                     method.token.line, method.token.column)
                if 'self' in method.parameters:
                    raise ParseError('Method self is implicit; omit it from parameters',
                                     method.token.line, method.token.column)
                if method.name == '__init__' and method.return_type not in (None, 'null'):
                    raise ParseError("Constructor '__init__' may only declare -> null",
                                     method.token.line, method.token.column)
                names.add(method.name)
                methods.append(method)
            self._expect('RBRACE', "'}' after methods")
            return ClassDeclaration(token, name.value, tuple(methods), parent)
        if self._match('USE'):
            module = self._expect('IDENT', "'gui', 'pandas', or 'numpy'")
            if module.value not in ('gui', 'pandas', 'numpy', 'pytorch'):
                raise self._error("Supported libraries: gui, pandas, numpy, pytorch")
            self._expect('DOT', "'.'")
            return UseGui(token) if module.value == 'gui' else UseLibrary(token, module.value)
        if self._words('save'):
            start = self.position
            self._take_words('save')
            try:
                table = self._postfix()
            except ParseError:
                self.position = start
            else:
                if self._words('as'):
                    self._take_words('as csv to')
                    path = self._postfix()
                    self._expect('DOT', "'.' after statement")
                    return SaveCsv(token, table, path)
                self.position = start
        if self._match('LET'):
            name = self._expect('IDENT', 'variable name')
            self._expect('BE', "'be'")
            result = Let(token, name.value, self._expression())
        elif self._match('SET'):
            target = self._postfix()
            if not isinstance(target, (Name, Index, Property)):
                raise self._error('Expected assignment target name, index, or property')
            self._expect('TO', "'to'")
            result = Assign(token, target, self._expression())
        elif self._match('SAY'):
            result = Say(token, self._expression())
        elif self._match('PRINT'):
            result = Print(token, self._expression())
        elif self._match('IF'):
            branches = []
            while True:
                condition = self._expression()
                self._expect('THEN', "'then'")
                branches.append((condition, self._block()))
                if not self._match('ELSE'):
                    return If(token, tuple(branches), None)
                if not self._match('IF'):
                    return If(token, tuple(branches), self._block())
        elif self._match('REPEAT'):
            count = self._expression()
            self._expect('TIMES', "'times'")
            return Repeat(token, count, self._loop_block())
        elif self._match('WHILE'):
            condition = self._expression()
            return While(token, condition, self._loop_block())
        elif self._match('FOR'):
            if self._match('EACH'):
                name = self._expect('IDENT', 'iteration variable name')
                self._expect('IN', "'in'")
                iterable = self._expression()
                return ForEach(token, name.value, iterable, self._loop_block())
            name = self._expect('IDENT', 'iteration variable name')
            self._expect('FROM', "'from'")
            start = self._expression()
            self._expect('TO', "'to'")
            end = self._expression()
            step = self._expression() if self._match('STEP') else None
            return CounterFor(token, name.value, start, end, step, self._loop_block())
        elif token.kind in {'BREAK', 'CONTINUE'}:
            if self.loop_depth == 0:
                raise self._error(f'{token.lexeme} is only valid inside a loop')
            self._match(token.kind)
            result = Break(token) if token.kind == 'BREAK' else Continue(token)
        elif self._match('DEFINE'):
            name = self._expect('IDENT', 'function name')
            self._expect('WITH', "'with'")
            self._expect('LPAREN', "'(' before parameters")
            parameters = []
            if self.current.kind != 'RPAREN':
                while True:
                    parameter = self._expect('IDENT', 'parameter name')
                    if parameter.value in parameters:
                        raise ParseError(f'Duplicate parameter {parameter.value!r}',
                                         parameter.line, parameter.column)
                    parameters.append(parameter.value)
                    if not self._match('COMMA'):
                        break
            self._expect('RPAREN', "')' after parameters")
            return_type = None
            if self._match('ARROW'):
                annotation = self.current
                if annotation.lexeme not in RETURN_TYPES:
                    raise self._error('Expected return type (int, float, string, bool, null, list, tensor)')
                self.position += 1
                return_type = annotation.lexeme
            previous_loop_depth = self.loop_depth
            self.loop_depth = 0
            self.function_depth += 1
            try:
                body = self._block()
            finally:
                self.function_depth -= 1
                self.loop_depth = previous_loop_depth
            return Define(token, name.value, tuple(parameters), body, return_type)
        elif self._match('RETURN'):
            if self.function_depth == 0:
                raise ParseError('return is only valid inside a function',
                                 token.line, token.column)
            result = Return(token, self._expression())
        elif self._match('ASK'):
            prompt = self._expression()
            self._expect('INTO', "'into'")
            name = self._expect('IDENT', 'input variable name')
            result = Ask(token, prompt, name.value)
        else:
            result = ExpressionStatement(token, self._expression())
        self._expect('DOT', "'.' after statement")
        return result

    def _loop_block(self) -> Block:
        self.loop_depth += 1
        try:
            return self._block()
        finally:
            self.loop_depth -= 1

    def _block(self) -> Block:
        token = self._expect('LBRACE', "'{' to start block")
        statements = []
        while self.current.kind not in {'RBRACE', 'EOF'}:
            statements.append(self._statement())
        self._expect('RBRACE', "'}' to close block")
        return Block(token, tuple(statements))

    def _expression(self) -> Expr:
        return self._binary(self._conjunction, ('OR',))

    def _binary(self, operand, kinds: tuple[str, ...]) -> Expr:
        left = operand()
        while (token := self._match(*kinds)) is not None:
            left = Binary(token, left, token.lexeme, operand())
        return left

    def _conjunction(self) -> Expr:
        return self._binary(self._negation, ('AND',))

    def _negation(self) -> Expr:
        if (token := self._match('NOT')) is not None:
            return Unary(token, 'not', self._negation())
        return self._comparison()

    def _comparison(self) -> Expr:
        left = self._sum()
        if (token := self._match('IS')) is None:
            return left
        words = ['is']
        if self._match('NOT'):
            words.append('not')
            self._expect('EQUAL', "'equal' after 'is not'")
            words.append('equal')
            self._expect('TO', "'to' after 'equal'")
            words.append('to')
        elif self._match('EQUAL'):
            self._expect('TO', "'to' after 'equal'")
            words.extend(('equal', 'to'))
        else:
            direction = self._match('GREATER', 'LESS')
            if direction is None:
                raise self._error("Expected 'equal', 'not equal', 'greater', or 'less' after 'is'")
            words.append(direction.lexeme)
            self._expect('THAN', "'than'")
            words.append('than')
            # Only consume OR when it begins the longer comparison phrase.
            if (self.current.kind == 'OR' and
                    self.tokens[self.position + 1].kind == 'EQUAL'):
                self._match('OR')
                self._match('EQUAL')
                self._expect('TO', "'to' after 'or equal'")
                words.extend(('or', 'equal', 'to'))
        result = Binary(token, left, ' '.join(words), self._sum())
        if self.current.kind == 'IS':
            raise self._error('Comparisons cannot chain; combine comparisons with and/or')
        return result

    def _sum(self) -> Expr:
        return self._binary(self._product, ('PLUS', 'MINUS'))

    def _product(self) -> Expr:
        return self._binary(self._unary, ('STAR', 'SLASH', 'PERCENT'))

    def _unary(self) -> Expr:
        if (token := self._match('PLUS', 'MINUS')) is not None:
            return Unary(token, token.lexeme, self._unary())
        return self._power()

    def _power(self) -> Expr:
        left = self._postfix()
        if (token := self._match('CARET')) is not None:
            return Binary(token, left, '^', self._unary())
        return left

    def _arguments(self, closing: str, description: str) -> tuple[Expr, ...]:
        arguments = []
        if self.current.kind != closing:
            arguments.append(self._expression())
            while self._match('COMMA'):
                arguments.append(self._expression())
        self._expect(closing, description)
        return tuple(arguments)

    def _property_dot(self) -> bool:
        # Adjacent .name distinguishes member access from a statement terminator.
        if self.current.kind != 'DOT' or self.position + 1 >= len(self.tokens):
            return False
        following = self.tokens[self.position + 1]
        return (following.kind in {'IDENT', 'FROM', 'STEP'} and following.line == self.current.line
                and following.column == self.current.column + 1)

    def _postfix(self) -> Expr:
        expression = self._primary()
        while True:
            if (token := self._match('LPAREN')) is not None:
                expression = Call(token, expression, self._arguments('RPAREN', "')'"))
            elif self._property_dot():
                token = self._expect('DOT', "'.'")
                name = self._expect('IDENT', 'property name')
                expression = Property(token, expression, name.value)
            elif (token := self._match('LBRACKET')) is not None:
                index = self._expression()
                self._expect('RBRACKET', "']'")
                expression = Index(token, expression, index)
            else:
                return expression

    def _words(self, phrase):
        words = phrase.split()
        tokens = self.tokens[self.position:self.position + len(words)]
        return len(tokens) == len(words) and all(
            t.lexeme == word and (t.kind == 'IDENT' or word in KEYWORDS)
            for t, word in zip(tokens, words))

    def _take_words(self, phrase):
        for word in phrase.split():
            if not self._words(word):
                raise self._error(f"Expected '{word}' in library phrase")
            self.position += 1

    def _table_primary(self):
        token = self.current
        def take(phrase):
            self._take_words(phrase)
        def atom():
            return self._postfix()
        def result(op, *args, comparison=''):
            return TableExpr(token, op, tuple(args), comparison)
        for direction in ('first', 'last'):
            if self._words(direction):
                start = self.position
                take(direction)
                try:
                    count = self._unary()
                except ParseError:
                    self.position = start
                    break
                if not self._words('rows of'):
                    self.position = start
                    break
                take('rows of')
                return result(direction, count, atom())
        if self._words('read csv from'):
            take('read csv from')
            return result('read', atom())
        if self._words('number of'):
            take('number of')
            if self._words('rows'):
                take('rows in')
                return result('nrows', atom())
            take('columns in')
            return result('ncols', atom())
        if self._words('columns of'):
            take('columns of')
            return result('columns', atom())
        for stat in ('average', 'total', 'minimum', 'maximum', 'count'):
            if self._words(stat + ' of column'):
                take(stat + ' of column')
                column = atom()
                take('in')
                return result(stat, column, atom())
        if self._words('column'):
            start = self.position
            take('column')
            try:
                column = atom()
            except ParseError:
                self.position = start
            else:
                if self._words('from'):
                    take('from')
                    return result('column', column, atom())
                self.position = start
        if self._words('rows of'):
            take('rows of')
            table = atom()
            take('where column')
            column = atom()
            comparison = next((p for p in sorted(COMPARISON_PHRASES, key=len, reverse=True)
                               if self._words(p)), None)
            if comparison is None:
                raise self._error("Expected English comparison starting with 'is'")
            take(comparison)
            return result('filter', table, column, atom(), comparison=comparison)
        if self._words('drop missing rows from'):
            take('drop missing rows from')
            return result('drop', atom())
        if self._words('fill missing in column'):
            take('fill missing in column')
            column = atom()
            take('of')
            table = atom()
            take('with')
            return result('fill', column, table, atom())
        if self._words('table from array'):
            take('table from array')
            value = atom()
            take('with columns')
            return result('from_array', value, atom())
        if self._words('table with columns'):
            take('table with columns')
            columns = atom()
            take('and rows')
            return result('table', columns, atom())
        return None

    def _array_primary(self):
        token = self.current
        def phrase(words):
            if self._words(words):
                self._take_words(words)
                return True
            return False
        def atom():
            return self._unary()
        def result(op, *args):
            return ArrayExpr(token, op, tuple(args))
        for words, op in (('array of zeros with shape', 'zeros'),
                          ('array of ones with shape', 'ones'),
                          ('random array with shape', 'random'),
                          ('array with', 'create')):
            if phrase(words):
                return result(op, atom())
        if phrase('array from'):
            value = atom()
            if phrase('to'):
                return result('range', value, atom())
            return result('create', value)
        for words, op in (('shape of', 'shape'), ('average of', 'mean'),
                          ('total of', 'sum'), ('minimum of', 'min'),
                          ('maximum of', 'max'), ('standard deviation of', 'std'),
                          ('square root of', 'sqrt'), ('absolute values of', 'absolute'),
                          ('sine of', 'sin'), ('cosine of', 'cos'),
                          ('exponential of', 'exp'), ('logarithm of', 'log')):
            if phrase(words):
                return result(op, atom())
        # Single contextual starters must leave ordinary names/calls untouched.
        if ((self._words('transpose') or self._words('reshape'))
                and self.tokens[self.position + 1].kind not in ('LPAREN', 'LBRACKET', 'PLUS', 'MINUS')):
            start = self.position
            op = self.current.lexeme
            self.position += 1
            try:
                value = atom()
            except ParseError:
                self.position = start
                return None
            if op == 'transpose':
                return result(op, value)
            self._take_words('to')
            return result(op, value, atom())
        if phrase('matrix multiply'):
            left = atom()
            self._take_words('by')
            return result('matmul', left, atom())
        for word, connector, op in (('add', 'to', 'add'),
                                     ('subtract', 'from', 'subtract'),
                                     ('multiply', 'by', 'multiply'),
                                     ('divide', 'by', 'divide')):
            if not self._words(word):
                continue
            start = self.position
            self.position += 1
            arrays = word == 'add' and phrase('arrays')
            try:
                left = atom()
            except ParseError:
                self.position = start
                return None
            if not self._words('and' if arrays else connector):
                self.position = start
                return None
            self._take_words('and' if arrays else connector)
            return result(op, left, atom())
        if phrase('slice of'):
            value = atom()
            self._take_words('from')
            start = atom()
            self._take_words('to')
            return result('slice', value, start, atom())
        return None

    def _torch_arguments(self):
        self._expect('LPAREN', "'(' before arguments")
        args, keywords = [], []
        if self.current.kind != 'RPAREN':
            while True:
                if self.current.kind in {'IDENT', 'FROM', 'STEP'} and self.tokens[self.position + 1].lexeme == 'as':
                    name = self.current.value
                    self.position += 1
                    self._take_words('as')
                    if name in [k for k, _ in keywords]:
                        raise self._error('Duplicate keyword argument ' + name)
                    keywords.append((name, self._expression()))
                else:
                    if keywords:
                        raise self._error('Positional arguments must precede keyword arguments')
                    args.append(self._expression())
                if not self._match('COMMA'):
                    break
        self._expect('RPAREN', "')' after arguments")
        return tuple(args), tuple(keywords)

    def _torch_primary(self):
        token = self.current
        def take(words):
            if self._words(words):
                self._take_words(words)
                return True
            return False
        def atom():
            return self._unary()
        def result(op, *args, keywords=()):
            return TorchExpr(token, op, tuple(args), keywords)
        def options():
            if take('with options'):
                args, kw = self._torch_arguments()
                if args:
                    raise self._error('Options require name as value')
                return kw
            return ()
        for name in LAYERS:
            if take(name + ' layer with'):
                args, kw = self._torch_arguments()
                return TorchExpr(token, 'layer:' + name, args, kw)
        for words, op in [('cuda available','cuda'), ('best available device','best'),
                          ('cuda device count','device_count'), ('cuda device name','device_name'),
                          ('mean squared error loss','mse'), ('cross entropy loss','cross_entropy'),
                          ('images to tensors','to_tensor')]:
            if take(words):
                return result(op)
        if take('tensor with') or take('tensor from'):
            value = atom()
            return result('tensor', value, keywords=options())
        for words, op in [('device of','device'), ('dtype of','dtype'),
                          ('parameters of','parameters'), ('scalar value of','item'),
                          ('tensor values of','tolist'), ('clear gradients of','zero_grad'),
                          ('calculate gradients from','backward'), ('update weights using','step'),
                          ('training mode for','train'), ('evaluation mode for','eval'),
                          ('resize images to','resize'), ('load tensor from','load_tensor'),
                          ('python module','module')]:
            if take(words):
                return result(op, atom())
        for words, op, connector in [('predict with','predict','using'), ('infer with','infer','using'),
                       ('move','move','to'), ('convert tensor','convert','to'),
                       ('save model weights of','save_weights','to'),
                       ('load model weights into','load_weights','from'), ('save tensor','save_tensor','to')]:
            # Preserve ordinary names such as move unless a phrase is complete.
            if self._words(words):
                start = self.position
                self._take_words(words)
                try:
                    value = atom()
                except ParseError:
                    self.position = start
                    continue
                if not take(connector):
                    self.position = start
                    continue
                return result(op, value, atom())
        for name in ('adam', 'sgd'):
            if take(name + ' optimizer for'):
                model = atom()
                self._take_words('with learning rate')
                rate = atom()
                return result(name, model, rate, keywords=options())
        if take('calculate loss using'):
            loss = atom()
            self._take_words('with')
            args, kw = self._torch_arguments()
            return result('loss', loss, *args, keywords=kw)
        if take('normalize images with') or take('compose transforms with'):
            op = 'normalize' if token.lexeme == 'normalize' else 'compose'
            args, kw = self._torch_arguments()
            return TorchExpr(token, op, args, kw)
        if take('load images from'):
            path = atom()
            transforms = (atom(),) if take('with transforms') else ()
            return result('images', path, *transforms)
        for name in ('mnist', 'cifar10'):
            if take('load ' + name + ' from'):
                path = atom()
                return result(name, path, keywords=options())
        if take('make batches from'):
            dataset = atom()
            self._take_words('with size')
            size = atom()
            return result('batches', dataset, size, keywords=options())
        if take('python member'):
            name = atom()
            self._take_words('of')
            return result('member', name, atom())
        if take('python call'):
            callee = atom()
            self._take_words('with')
            args, kw = self._torch_arguments()
            return result('call', callee, *args, keywords=kw)
        if self._words('columns') and self.tokens[self.position + 1].kind == 'LBRACKET':
            self._take_words('columns')
            names = atom()
            self._take_words('from')
            return result('columns', names, atom())
        return None

    def _primary(self) -> Expr:
        ml = self._torch_primary()
        if ml is not None:
            return ml
        table = self._table_primary()
        if table is not None:
            return table
        array = self._array_primary()
        if array is not None:
            return array
        if (token := self._match('NUMBER', 'STRING', 'TRUE', 'FALSE', 'NULL')) is not None:
            return Literal(token, token.value)
        if (token := self._match('SUPER')) is not None:
            self._expect('LPAREN', "'(' after super")
            self._expect('RPAREN', "')' after super; super expects 0 arguments")
            return Super(token)
        if (token := self._match('IDENT')) is not None:
            return Name(token, token.value)
        if self._match('LPAREN'):
            expression = self._expression()
            self._expect('RPAREN', "')'")
            return expression
        if (token := self._match('LBRACKET')) is not None:
            return ListLiteral(token, self._arguments('RBRACKET', "']'"))
        raise self._error('Expected expression')


class _ReturnSignal(Exception):
    """Internal transfer caught only by the current function call."""

    def __init__(self, value: Any, node: Return):
        self.value = value
        self.node = node


class _BreakSignal(Exception):
    """Internal transfer caught by the nearest executing loop."""


class _ContinueSignal(Exception):
    """Internal transfer caught by the nearest executing loop."""


@dataclass(eq=False)
class GuiHandle:
    owner: Any
    widget: Any
    kind: str
    window: Any = None
    closed: bool = False

    def __str__(self):
        return '<gui ' + self.kind + '>'


class PythonBridge:
    """Internal allowlisted library bridge. Objects stay alive as ordinary bindings.

    Adapters share lazy imports and located error translation. PyTorch exposes
    explicit allowlisted module/member/call interoperability; no Python
    source evaluation is used. This bridge is not a security sandbox.
    """
    LIBRARIES = frozenset({'pandas', 'numpy', 'pytorch'})

    def __init__(self):
        self.modules = {}
        self.instances = {}
        self.callables = {}

    def enable(self, name, node):
        if name not in self.LIBRARIES:
            Interpreter._error(node, f'Unsupported library {name!r}')
        if name not in self.modules:
            import importlib
            import sys
            try:
                self.modules[name] = importlib.import_module('torch' if name == 'pytorch' else name)
            except Exception as error:
                Interpreter._error(node, f"Cannot load {name}: {error}. "
                                   + ("PyTorch is not installed in the Python environment used by Vexa. " if name == "pytorch" else "")
                                   + f"Install into this Python with "
                                   f'"{sys.executable}" -m pip install {"torch torchvision" if name == "pytorch" else name}')
        return self.modules[name]

    def require(self, name, node):
        if name not in self.modules:
            Interpreter._error(node, f'Library operations require use {name}. first')
        return self.modules[name]

    @staticmethod
    def convert(value, node):
        # Only plain finite scalar/list inputs cross from Vexa to library adapters.
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value) is list:
            return [PythonBridge.convert(v, node) for v in value]
        Interpreter._error(node, 'Library input must contain only strings, numbers, booleans, null, or lists')

    @staticmethod
    def call(node, description, action):
        try:
            return action()
        except VexaError:
            raise
        except Exception as error:
            Interpreter._error(node, f'{description}: {error}')


class Interpreter:
    """Tree-walking runtime for the supported Vexa AST."""

    def __init__(self, environment: Environment | None = None, output=None, input_stream=None):
        import sys
        builtins = Environment()
        for name, (parameters, _) in BUILTIN_METADATA.items():
            arity = len(parameters)
            builtins.values[name] = Builtin(name, arity)
        self._builtins = builtins
        self._modules: dict[Path, Module] = {}
        self._active_modules: list[Path] = []
        self.environment = environment if environment is not None else Environment(builtins)
        if environment is not None:
            root = environment
            while root.parent is not None:
                root = root.parent
            root.parent = builtins
        self.input_stream = input_stream if input_stream is not None else sys.stdin
        self.output = output if output is not None else sys.stdout
        self.loop_depth = 0
        self.function_depth = 0
        self.bridge = PythonBridge()
        self.gui_enabled = False
        self._tk = None
        self._windows = []

    @staticmethod
    def _error(node: Node, message: str):
        raise VexaRuntimeError(message, node.token.line, node.token.column, node.token.source_path)

    @staticmethod
    def _number(value: Any) -> bool:
        return type(value) in (int, float)

    def _boolean(self, value: Any, node: Node) -> bool:
        if type(value) is not bool:
            self._error(node, 'Logical operands must be booleans')
        return value

    @staticmethod
    def display(value: Any, seen=None) -> str:
        if value is None:
            return 'null'
        if type(value) is bool:
            return 'true' if value else 'false'
        if isinstance(value, Module):
            return f"<module {value.path.stem}>"
        if isinstance(value, ClassValue):
            return f"<class {value.name}>"
        if isinstance(value, Instance):
            return f"<{value.klass.name} instance>"
        if isinstance(value, (Function, Builtin, BoundMethod)):
            return '<function>'
        if type(value) is list:
            seen = set() if seen is None else seen
            if id(value) in seen:
                return '[...]'
            seen.add(id(value))
            result = '[' + ', '.join(json.dumps(v, ensure_ascii=False) if type(v) is str
                                    else Interpreter.display(v, seen) for v in value) + ']'
            seen.remove(id(value))
            return result
        return str(value)

    def _index(self, collection, index, node):
        torch = self.bridge.modules.get('pytorch')
        if torch is not None and isinstance(collection, torch.Tensor):
            if type(index) is not int or index < 0 or collection.ndim == 0 or index >= len(collection):
                self._error(node, 'Tensor index must be a nonnegative integer within its shape')
            return index
        np = self.bridge.modules.get('numpy')
        if np is not None and isinstance(collection, np.ndarray):
            if type(index) is not int or index < 0:
                self._error(node, 'Array index must be a nonnegative integer')
            if collection.ndim == 0 or index >= collection.shape[0]:
                self._error(node, 'Array index out of range')
            return index
        if type(collection) is not list:
            self._error(node, 'Indexing requires a list or array')
        if type(index) is not int:
            self._error(node, 'List index must be an integer')
        if not 0 <= index < len(collection):
            self._error(node, 'List index out of range')
        return index

    def _array(self, node):
        args = [self.evaluate(arg) for arg in node.arguments]
        torch = self.bridge.modules.get('pytorch')
        if torch is not None and any(isinstance(v, torch.Tensor) for v in args):
            return self._torch_array(node, args)
        np = self.bridge.require('numpy', node)
        op = node.operation
        def shape(value):
            if type(value) is not list or any(type(n) is not int or n < 0 for n in value):
                self._error(node, 'Shape must be a list of nonnegative integers')
            return tuple(value)
        def array(value):
            if not isinstance(value, np.ndarray):
                self._error(node, 'Expected a NumPy array')
            return value
        def numeric(value):
            if isinstance(value, np.ndarray):
                if value.dtype.kind not in 'iuf':
                    self._error(node, 'Numerical operations require numeric arrays; text, booleans, and objects are not numbers')
                return value
            if self._number(value):
                return value
            self._error(node, 'Expected a numeric array or number')
        def operation():
            if op == 'create':
                value = args[0]
                if isinstance(value, np.ndarray):
                    return value
                pd = self.bridge.modules.get('pandas')
                if pd is not None and isinstance(value, (pd.Series, pd.DataFrame)):
                    return value.to_numpy(copy=False)
                if type(value) is not list:
                    self._error(node, 'Array creation requires a list, array, pandas column, or table')
                value = self.bridge.convert(value, node)
                # Object dtype prevents mixed text/numbers from being coerced to strings.
                def kinds(v):
                    return set().union(*(kinds(x) for x in v)) if type(v) is list else {type(v)}
                def dimensions(v):
                    if type(v) is not list:
                        return ()
                    children = [dimensions(x) for x in v]
                    if children and any(d != children[0] for d in children):
                        self._error(node, 'Array lists must be rectangular, with matching nested shapes')
                    return (len(v),) + (children[0] if children else ())
                dimensions(value)
                types = kinds(value)
                mixed = not (types <= {int, float} or len(types) == 1)
                return np.asarray(value, dtype=object if mixed else None)
            if op in ('zeros', 'ones', 'random'):
                dimensions = shape(args[0])
                return np.random.random(dimensions) if op == 'random' else getattr(np, op)(dimensions)
            if op == 'range':
                if any(type(v) is not int for v in args):
                    self._error(node, 'Array range endpoints must be integers')
                if args[0] > args[1]:
                    self._error(node, 'Array range start must not exceed its end')
                return np.arange(args[0], args[1] + 1)
            value = array(args[0]) if op not in ('add', 'subtract', 'multiply', 'divide') else args[0]
            if op == 'shape':
                return list(value.shape)
            if op == 'reshape':
                return np.reshape(value, shape(args[1]))
            if op == 'transpose':
                return np.transpose(value)
            if op == 'slice':
                if value.ndim == 0:
                    self._error(node, 'Cannot slice a zero-dimensional array')
                if any(type(v) is not int or v < 0 for v in args[1:]):
                    self._error(node, 'Slice bounds must be nonnegative integers')
                return value[args[1]:args[2]]
            numeric(value)
            with np.errstate(all='raise'):
                if op in ('add', 'subtract', 'multiply', 'divide', 'matmul'):
                    if not any(isinstance(v, np.ndarray) for v in args):
                        self._error(node, 'Array arithmetic requires at least one array')
                    left, right = (numeric(v) for v in args)
                    if op == 'subtract':
                        left, right = right, left
                    result = getattr(np, op)(left, right)
                else:
                    result = getattr(np, op)(value)
            return result.item() if isinstance(result, np.generic) else result
        return self.bridge.call(node, 'Array ' + op + ' failed', operation)

    def _batch_iterator(self, values, node):
        iterator = self.bridge.call(node, 'Cannot read batches', lambda: iter(values))
        while True:
            try:
                value = next(iterator)
            except StopIteration:
                return
            except Exception as error:
                self._error(node, f'Cannot read batch: {error}')
            # Default collation returns a list; custom tuple batches remain accessible.
            yield list(value) if type(value) is tuple else value

    def _torch_array(self, node, args):
        t = self.bridge.require('pytorch', node)
        op = node.operation
        def operation():
            a = args[0]
            if op == 'shape': return list(a.shape)
            if op == 'reshape': return a.reshape(args[1])
            if op == 'transpose': return a.permute(*reversed(range(a.ndim)))
            if op == 'slice':
                if any(type(x) is not int or x < 0 for x in args[1:]):
                    self._error(node, 'Slice bounds must be nonnegative integers')
                return a[args[1]:args[2]]
            if op in ('add', 'subtract', 'multiply', 'divide', 'matmul'):
                a, b = args
                if op == 'subtract': a, b = b, a
                return {'add': t.add, 'subtract': t.sub, 'multiply': t.mul,
                        'divide': t.div, 'matmul': t.matmul}[op](a, b)
            if op == 'std': return t.std(a, correction=0)
            if op == 'mean' and not a.is_floating_point(): a = a.float()
            return getattr(t, {'absolute': 'abs'}.get(op, op))(a)
        return self.bridge.call(node, 'Tensor ' + op + ' failed (check shapes and data types)', operation)

    def _torch(self, node):
        t = self.bridge.require('pytorch', node)
        args = [self.evaluate(a) for a in node.arguments]
        kw = {k: self.evaluate(v) for k, v in node.keywords}
        op = node.operation
        def device(value):
            d = t.device(value)
            if d.type == 'cuda' and not t.cuda.is_available():
                self._error(node, 'CUDA is not available on this computer.')
            return d
        def dtype(value):
            allowed = ('float16', 'float32', 'float64', 'bfloat16', 'int8', 'int16', 'int32', 'int64', 'uint8', 'bool')
            if value not in allowed:
                self._error(node, 'Tensor dtype must be one of: ' + ', '.join(allowed))
            return getattr(t, value)
        def vision():
            import importlib
            return importlib.import_module('torchvision')
        def operation():
            if op.startswith('layer:'):
                return getattr(t.nn, LAYERS[op.split(':')[1]])(*args, **kw)
            if op == 'cuda': return t.cuda.is_available()
            if op == 'best': return t.device('cuda' if t.cuda.is_available() else 'cpu')
            if op == 'device_count': return t.cuda.device_count()
            if op == 'device_name':
                device('cuda')
                return t.cuda.get_device_name()
            if op == 'tensor':
                if set(kw) - {'dtype', 'device', 'requires_grad'}:
                    self._error(node, 'Tensor options: dtype, device, requires_grad')
                if 'dtype' in kw: kw['dtype'] = dtype(kw['dtype'])
                if 'device' in kw: kw['device'] = device(kw['device'])
                value = args[0]
                pd = self.bridge.modules.get('pandas')
                if pd is not None and isinstance(value, (pd.Series, pd.DataFrame)):
                    value = value.to_numpy(copy=True)
                # Validate before conversion so strings and missing data never disappear.
                import numpy as np
                if not isinstance(value, t.Tensor):
                    arr = np.asarray(value)
                    if arr.dtype.kind not in 'biuf':
                        self._error(node, 'This column contains text or missing objects but a numeric tensor was expected.')
                    if not np.isfinite(arr).all():
                        self._error(node, 'Tensor data contains missing or nonfinite values; fill or drop missing rows explicitly.')
                    # Native lists use PyTorch dtype inference (float32 by default);
                    # NumPy/pandas preserve their numeric dtype, with owned storage.
                    value = value if type(value) is list else arr.copy()
                    result = t.tensor(value, **{k:v for k,v in kw.items() if k != 'requires_grad'})
                else:
                    result = value.to(**{k:v for k,v in kw.items() if k != 'requires_grad'})
                if 'requires_grad' in kw:
                    if type(kw['requires_grad']) is not bool:
                        self._error(node, 'requires_grad must be true or false')
                    result.requires_grad_(kw['requires_grad'])
                return result
            if op == 'device': return args[0].device
            if op == 'dtype': return str(args[0].dtype)
            if op == 'convert': return args[0].to(dtype=dtype(args[1]))
            if op == 'move': return args[0].to(device(args[1]))
            if op == 'parameters': return list(args[0].parameters())
            if op == 'item': return args[0].item()
            if op == 'tolist': return args[0].detach().cpu().tolist()
            if op in ('predict', 'infer'):
                model, inputs = args
                if not isinstance(model, t.nn.Module) or not isinstance(inputs, t.Tensor):
                    self._error(node, 'Prediction requires a model and an input tensor')
                if any(p.device != inputs.device for p in list(model.parameters()) + list(model.buffers())):
                    self._error(node, 'The model and input tensor are on different devices.')
                if op == 'infer':
                    with t.no_grad(): return model(inputs)
                return model(inputs)
            if op == 'mse': return t.nn.MSELoss()
            if op == 'cross_entropy': return t.nn.CrossEntropyLoss()
            if op == 'loss':
                if len(args) != 3 or kw:
                    self._error(node, 'Calculate loss needs exactly (predictions, targets)')
                criterion, predictions, targets = args
                if isinstance(criterion, t.nn.MSELoss) and predictions.shape != targets.shape:
                    self._error(node, 'Mean squared error requires predictions and targets with identical shapes.')
                if isinstance(criterion, t.nn.CrossEntropyLoss) and targets.dtype != t.int64 and targets.shape != predictions.shape:
                    self._error(node, 'Cross entropy class targets must use int64; probability targets must match prediction shape.')
                return criterion(predictions, targets)
            if op in ('adam', 'sgd'):
                return (t.optim.Adam if op == 'adam' else t.optim.SGD)(args[0].parameters(), lr=args[1], **kw)
            if op in ('zero_grad', 'backward', 'step', 'train', 'eval'):
                return getattr(args[0], op)()
            if op in ('save_weights', 'save_tensor'):
                path = self._local_path(args[1], node)
                if op == 'save_tensor' and not isinstance(args[0], t.Tensor):
                    self._error(node, 'save tensor requires a tensor')
                t.save(args[0].state_dict() if op == 'save_weights' else args[0], path)
                return None
            if op in ('load_weights', 'load_tensor'):
                path = self._local_path(args[-1], node)
                loaded = t.load(path, map_location='cpu', weights_only=True)
                if op == 'load_weights':
                    args[0].load_state_dict(loaded)
                    return None
                if not isinstance(loaded, t.Tensor):
                    self._error(node, 'The file does not contain tensor data')
                return loaded
            if op == 'columns':
                table = self._table_value(args[1], node)
                if type(args[0]) is not list or any(type(n) is not str for n in args[0]):
                    self._error(node, 'Column names must be a list of strings')
                return table[args[0]].copy()
            if op in ('resize', 'to_tensor', 'normalize', 'compose'):
                transforms = vision().transforms
                if op == 'resize': return transforms.Resize(*args, **kw)
                if op == 'to_tensor': return transforms.ToTensor()
                if op == 'normalize': return transforms.Normalize(*args, **kw)
                return transforms.Compose(list(args))
            if op == 'images':
                from pathlib import Path
                path = self._local_path(args[0], node)
                if not Path(path).is_dir(): self._error(node, 'The dataset folder was not found.')
                v = vision()
                return v.datasets.ImageFolder(path, transform=args[1] if len(args) == 2 else v.transforms.ToTensor())
            if op in ('mnist', 'cifar10'):
                v = vision()
                kw.setdefault('download', False)
                kw.setdefault('transform', v.transforms.ToTensor())
                if type(kw['download']) is not bool:
                    self._error(node, 'download must be explicitly true or false')
                return getattr(v.datasets, 'MNIST' if op == 'mnist' else 'CIFAR10')(self._local_path(args[0], node), **kw)
            if op == 'batches':
                if type(args[1]) is not int or args[1] <= 0:
                    self._error(node, 'Batch size must be a positive integer')
                return t.utils.data.DataLoader(args[0], batch_size=args[1], **kw)
            if op == 'module':
                allowed = ('torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils.data',
                           'torchvision.datasets', 'torchvision.transforms', 'torchvision.models')
                if args[0] not in allowed: self._error(node, 'Python module must be an allowed PyTorch module')
                import importlib
                return importlib.import_module(args[0])
            if op == 'member':
                if type(args[0]) is not str or args[0].startswith('_') or '.' in args[0]:
                    self._error(node, 'Python member must be a single public name')
                import types
                allowed = {
                    'torch': {'tensor', 'zeros', 'ones', 'randn', 'relu', 'sigmoid', 'tanh'},
                    'torch.nn': set(LAYERS.values()) | {'Module', 'Parameter', 'MSELoss', 'CrossEntropyLoss'},
                    'torch.nn.functional': {'relu', 'sigmoid', 'tanh', 'linear', 'mse_loss', 'cross_entropy'},
                    'torch.optim': {'Adam', 'SGD'},
                    'torch.utils.data': {'TensorDataset', 'DataLoader'},
                    'torchvision.transforms': {'Resize', 'ToTensor', 'Normalize', 'Compose'},
                    'torchvision.datasets': {'ImageFolder', 'MNIST', 'CIFAR10'},
                    'torchvision.models': set(),
                }
                module = args[1]
                if not isinstance(module, types.ModuleType) or args[0] not in allowed.get(module.__name__, set()):
                    self._error(node, 'Python member is not an allowlisted PyTorch API')
                result = getattr(module, args[0])
                self.bridge.callables[id(result)] = result
                return result
            if op == 'call':
                # Only values issued by the member allowlist or real modules are callable.
                if not isinstance(args[0], t.nn.Module) and id(args[0]) not in self.bridge.callables:
                    self._error(node, 'Python call requires an allowlisted PyTorch API')
                return args[0](*args[1:], **kw)
            self._error(node, 'Unknown PyTorch operation ' + op)
        return self.bridge.call(node, 'PyTorch ' + op + ' failed (check shapes, types and devices)', operation)

    def _table_value(self, value, node):
        pd = self.bridge.require('pandas', node)
        if not isinstance(value, pd.DataFrame):
            self._error(node, 'Expected a table')
        return value

    def _table_column(self, table, column, source, node):
        if type(column) is not str:
            self._error(node, 'Column name must be a string')
        if column not in table.columns:
            label = f" '{source.identifier}'" if isinstance(source, Name) else ''
            self._error(node, f"Column {column!r} was not found in table{label}.")
        return table[column]

    def _local_path(self, value, node):
        from pathlib import Path
        if type(value) is not str or not value or '://' in value:
            self._error(node, 'CSV path must be a nonempty local file path string')
        return Path(value)

    def _save_csv(self, node):
        self.bridge.require('pandas', node)
        table = self._table_value(self.evaluate(node.table), node)
        path = self._local_path(self.evaluate(node.path), node)
        def save():
            # Exclusive creation avoids check-then-write overwrite races.
            with path.open('x', encoding='utf-8', newline='') as stream:
                table.to_csv(stream, index=False)
        if path.exists():
            self._error(node, f'CSV output file already exists: {path}. Choose a new filename')
        return self.bridge.call(node, 'Could not save CSV file', save)

    def _table(self, node):
        pd = self.bridge.require('pandas', node)
        args = [self.evaluate(arg) for arg in node.arguments]
        op = node.operation
        def operation():
            if op == 'read':
                path = self._local_path(args[0], node)
                # Open ourselves so pandas cannot interpret URLs or compressed paths.
                with path.open('r', encoding='utf-8', newline='') as stream:
                    return pd.read_csv(stream)
            if op == 'from_array':
                np = self.bridge.require('numpy', node)
                value, columns = args
                if not isinstance(value, np.ndarray) or value.ndim != 2:
                    self._error(node, 'Table conversion requires a two-dimensional array')
                if (type(columns) is not list or any(type(c) is not str for c in columns)
                        or len(set(columns)) != len(columns) or len(columns) != value.shape[1]):
                    self._error(node, 'Provide one unique string column name per array column')
                return pd.DataFrame(value, columns=columns, copy=False)
            if op == 'table':
                columns, rows = self.bridge.convert(args, node)
                if (type(columns) is not list or any(type(c) is not str for c in columns)
                        or len(set(columns)) != len(columns)):
                    self._error(node, 'Table columns must be a list of unique strings')
                if type(rows) is not list or any(type(r) is not list or len(r) != len(columns) for r in rows):
                    self._error(node, 'Each table row must be a list with one cell per column')
                if any(type(cell) is list for row in rows for cell in row):
                    self._error(node, 'Table cells must be scalar values, not lists')
                return pd.DataFrame(rows, columns=columns)
            table_index = 1 if op in ('first', 'last', 'column', 'average', 'total', 'minimum', 'maximum', 'count', 'fill') else 0
            table = self._table_value(args[table_index], node)
            if op in ('first', 'last'):
                if type(args[0]) is not int or args[0] < 0:
                    self._error(node, 'Row count must be a nonnegative integer')
                return (table.head(args[0]) if op == 'first' else table.tail(args[0])).copy()
            if op == 'nrows':
                return len(table)
            if op == 'ncols':
                return len(table.columns)
            if op == 'columns':
                return list(table.columns)
            if op == 'drop':
                return table.dropna().copy()
            column_index = 1 if op == 'filter' else 0
            series = self._table_column(table, args[column_index], node.arguments[table_index], node)
            if op == 'column':
                return series.copy()
            if op == 'count':
                return int(series.count())
            if op == 'fill':
                replacement = self.bridge.convert(args[2], node)
                if replacement is None or type(replacement) is list:
                    self._error(node, 'Fill value must be a nonmissing scalar')
                result = table.copy()
                result[args[0]] = series.fillna(replacement)
                return result
            values = series.dropna().tolist()
            if op == 'filter':
                value = self.bridge.convert(args[2], node)
                if type(value) is list or value is None:
                    self._error(node, 'Filter comparison requires a nonmissing scalar')
                def compatible(cell):
                    return (self._number(cell) and self._number(value)) or type(cell) is type(value)
                if not all(compatible(cell) for cell in values):
                    self._error(node, 'Filter values must have matching types; numbers are not text')
                if node.comparison not in ('is equal to', 'is not equal to') and not (
                        self._number(value) or type(value) is str):
                    self._error(node, 'Filter ordering requires numbers or strings')
                mask = {'is equal to': series.eq, 'is not equal to': series.ne,
                        'is greater than': series.gt, 'is less than': series.lt,
                        'is greater than or equal to': series.ge,
                        'is less than or equal to': series.le}[node.comparison](value)
                return table.loc[series.notna() & mask.fillna(False)].copy()
            if not all(self._number(cell) for cell in values):
                self._error(node, f'{op} requires numeric column values; text and booleans are not numbers')
            if not values:
                return 0 if op == 'total' else None
            value = getattr(series, {'average': 'mean', 'total': 'sum', 'minimum': 'min', 'maximum': 'max'}[op])()
            value = value.item() if hasattr(value, 'item') else value
            if not self._number(value) or (type(value) is float and not math.isfinite(value)):
                self._error(node, 'Table statistic produced a nonfinite or unsupported number')
            return value
        return self.bridge.call(node, 'Table ' + op + ' failed', operation)

    def _gui(self, name, args, node):
        if not self.gui_enabled:
            self._error(node, 'GUI requires use gui. before GUI calls')

        def integer(value, label, positive=False):
            if type(value) is not int or (positive and value <= 0):
                self._error(node, label + ' must be a ' + ('positive ' if positive else '') + 'integer')

        def text(value):
            if type(value) is not str:
                self._error(node, 'GUI text and colors must be strings')

        def handle(value, window=False):
            if not isinstance(value, GuiHandle) or value.owner is not self:
                self._error(node, 'Expected a GUI widget from this program')
            if window and value.kind != 'window':
                self._error(node, 'Expected a GUI window')
            root = value if value.kind == 'window' else value.window
            if root.closed:
                self._error(node, 'GUI window is closed')
            return value

        try:
            if name == 'make_window':
                text(args[0])
                integer(args[1], 'Window width', True)
                integer(args[2], 'Window height', True)
                if self._tk is None:
                    import tkinter
                    self._tk = tkinter
                parent = next((w for w in self._windows if not w.closed), None)
                widget = self._tk.Toplevel(parent.widget) if parent else self._tk.Tk()
                result = GuiHandle(self, widget, 'window')
                try:
                    widget.title(args[0])
                    widget.geometry(f'{args[1]}x{args[2]}')
                    def destroyed(event):
                        if event.widget is widget:
                            result.closed = True
                    widget.bind('<Destroy>', destroyed, add='+')
                except Exception:
                    widget.destroy()
                    raise
                self._windows.append(result)
                return result
            target = handle(args[0], name in ('make_button', 'make_label', 'make_text_input',
                                             'show_window', 'close_window'))
            widget = target.widget
            if name in ('make_button', 'make_label', 'make_text_input'):
                options = {}
                if name != 'make_text_input':
                    text(args[1])
                    options['text'] = args[1]
                if name == 'make_button':
                    callback = args[2]
                    if not isinstance(callback, Function) or callback.declaration.parameters:
                        self._error(node, 'Button callback must be a Vexa function with zero parameters')
                    def click():
                        try:
                            self._invoke(callback, [], node)
                        except VexaError as error:
                            print(f'GUI callback error: {error}', file=self.output)
                        except Exception as error:
                            print(f'GUI callback error: {type(error).__name__}: {error}', file=self.output)
                    options['command'] = click
                kind = {'make_button': 'button', 'make_label': 'label', 'make_text_input': 'input'}[name]
                constructor = {'button': self._tk.Button, 'label': self._tk.Label, 'input': self._tk.Entry}[kind]
                return GuiHandle(self, constructor(widget, **options), kind, target)
            if name == 'get_text':
                if target.kind == 'window':
                    return widget.title()
                return widget.get() if target.kind == 'input' else widget.cget('text')
            if name == 'set_text':
                text(args[1])
                if target.kind == 'window':
                    widget.title(args[1])
                elif target.kind == 'input':
                    widget.delete(0, 'end')
                    widget.insert(0, args[1])
                else:
                    widget.configure(text=args[1])
            elif name == 'place':
                if target.kind == 'window':
                    self._error(node, 'place requires a child widget')
                for label, value in zip(('x', 'y', 'width', 'height'), args[1:]):
                    integer(value, label, label in ('width', 'height'))
                widget.place(x=args[1], y=args[2], width=args[3], height=args[4])
            elif name == 'style':
                options = {}
                for key, value in zip(('background', 'foreground'), args[1:3]):
                    if value is not None:
                        text(value)
                        options[key] = value
                if args[3] is not None:
                    integer(args[3], 'Font size', True)
                    options['font'] = ('TkDefaultFont', args[3])
                if target.kind == 'window' and any(key != 'background' for key in options):
                    self._error(node, 'Windows support background styling only')
                widget.configure(**options)
            elif name == 'show_window':
                widget.wait_window(widget)
            elif name == 'close_window':
                widget.destroy()
                target.closed = True
        except VexaError:
            raise
        except (ImportError, OSError) as error:
            self._error(node, f'GUI unavailable: {error}. Install Tkinter and use a desktop display')
        except Exception as error:
            self._error(node, f'{name}: GUI operation failed: {error}')
        return None

    def _builtin(self, function, args, node):
        name = function.name
        if len(args) != function.arity:
            self._error(node, f'{name} expects {function.arity} arguments, got {len(args)}')
        if name in GUI_METADATA:
            return self._gui(name, args, node)
        try:
            if name == 'to_string':
                return self.display(args[0])
            if name in ('to_int', 'to_float'):
                value = args[0]
                if type(value) is not str:
                    self._error(node, f'{name} requires a string')
                pattern = r'[+-]?[0-9]+' if name == 'to_int' else r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
                if not re.fullmatch(pattern, value.strip()):
                    self._error(node, f'Invalid numeric input for {name}')
                result = int(value) if name == 'to_int' else float(value)
            elif name == 'length':
                if type(args[0]) not in (str, list):
                    self._error(node, 'length requires a string or list')
                return len(args[0])
            elif name in ('append', 'pop'):
                if type(args[0]) is not list:
                    self._error(node, f'{name} requires a list')
                if name == 'append':
                    args[0].append(args[1])
                    return None
                if not args[0]:
                    self._error(node, 'Cannot pop an empty list')
                return args[0].pop()
            elif name == 'random':
                if any(type(v) is not int for v in args):
                    self._error(node, 'random requires integer bounds')
                if args[0] > args[1]:
                    self._error(node, 'random lower bound must not exceed upper bound')
                return random_module.randint(*args)
            else:
                if not all(self._number(v) for v in args):
                    self._error(node, f'{name} requires numbers')
                result = {'abs': abs, 'sqrt': math.sqrt, 'floor': math.floor,
                          'ceil': math.ceil, 'round': round, 'min': min, 'max': max}[name](*args)
        except (ValueError, OverflowError):
            self._error(node, f'{name}: invalid input or numeric result out of range')
        if type(result) is float and not math.isfinite(result):
            self._error(node, f'{name}: numeric result is not finite')
        return result

    @staticmethod
    def _equal(left, right, seen=None):
        if type(left) not in (list, str, bool, int, float, type(None)):
            return left is right
        if type(left) is list and type(right) is list:
            seen = set() if seen is None else seen
            pair = (id(left), id(right))
            if pair in seen:
                return True
            seen.add(pair)
            return len(left) == len(right) and all(
                Interpreter._equal(a, b, seen) for a, b in zip(left, right))
        return (type(left) is type(right) or
                (Interpreter._number(left) and Interpreter._number(right))) and left == right

    def _load_module(self, parts: tuple[str, ...], statement: Import) -> Module:
        source_path = statement.token.source_path
        directory = source_path.parent if source_path is not None else Path.cwd()
        name = '.'.join(parts)
        candidate = directory.joinpath(*parts).with_suffix('.vexa')
        try:
            path = candidate.resolve()
        except (OSError, RuntimeError) as error:
            self._error(statement, f'Cannot resolve module {name!r} at {candidate}: {error}')
        if path in self._active_modules:
            chain = ' -> '.join(str(active) for active in (*self._active_modules, path))
            self._error(statement, f'Circular import: {chain}')
        if path in self._modules:
            return self._modules[path]
        try:
            source = path.read_text(encoding='utf-8')
        except FileNotFoundError:
            self._error(statement, f'Module {name!r} not found at {candidate}')
        except (OSError, UnicodeError) as error:
            self._error(statement, f'Cannot read module {name!r} at {candidate}: {error}')

        previous = self.environment
        module = Module(path, Environment(self._builtins))
        try:
            program = Parser(Lexer(source, path).tokenize()).parse()
            self.environment = module.environment
            self.execute(program)
        except VexaError as error:
            if error.source_path is None:
                error.source_path = path
            raise
        finally:
            self.environment = previous
        self._modules[path] = module
        return module

    def _module_member(self, module: Module, name: str, node: Node) -> Any:
        if name not in module.environment.values:
            self._error(node, f'Module {module.path.stem!r} has no member {name!r}')
        return module.environment.values[name]

    def _execute_import(self, statement: Import) -> None:
        # Check every destination before loading modules, and bind only after all lookups succeed.
        names = set(self.environment.values)
        for _, name in statement.bindings:
            if name in names:
                self._error(statement, f'Duplicate binding {name!r}')
            names.add(name)
        if statement.module is None:
            values = {name: self._load_module(path, statement)
                      for path, name in statement.bindings}
        else:
            module = self._load_module(statement.module, statement)
            values = {name: self._module_member(module, path[0], statement)
                      for path, name in statement.bindings}
        self.environment.values.update(values)

    def execute(self, program: Program) -> None:
        path = program.token.source_path
        if path is not None:
            self._active_modules.append(path)
        try:
            for statement in program.statements:
                try:
                    self.execute_statement(statement)
                except RecursionError:
                    self._error(statement, 'Expression or value nesting exceeds runtime recursion limit')
        finally:
            if path is not None:
                self._active_modules.pop()

    def execute_block(self, block: Block) -> None:
        # The language contract gives blocks their enclosing lexical scope.
        for statement in block.statements:
            self.execute_statement(statement)

    def _iteration(self, body: Block) -> bool:
        """Execute a body; return false on break, true on completion/continue."""
        self.loop_depth += 1
        try:
            self.execute_block(body)
        except _BreakSignal:
            return False
        except _ContinueSignal:
            pass
        finally:
            self.loop_depth -= 1
        return True

    def execute_statement(self, statement: Stmt) -> None:
        if isinstance(statement, Import):
            self._execute_import(statement)
        elif isinstance(statement, UseLibrary):
            self.bridge.enable(statement.name, statement)
        elif isinstance(statement, SaveCsv):
            self._save_csv(statement)
        elif isinstance(statement, UseGui):
            self.gui_enabled = True
        elif isinstance(statement, Let):
            # Reject duplicates before evaluating their initializer.
            if statement.name in self.environment.values:
                self._error(statement, f'Duplicate binding {statement.name!r}')
            self.environment.define(statement.name, self.evaluate(statement.value),
                                    statement.token)
        elif isinstance(statement, Assign):
            target = statement.target
            if isinstance(target, Name):
                owner = self.environment._owner(target.identifier, target.token)
                owner.assign(target.identifier, self.evaluate(statement.value), target.token)
            elif isinstance(target, Property):
                instance = self.evaluate(target.object)
                if isinstance(instance, Module):
                    self._error(target, 'Module namespace bindings are read-only')
                instance = self.bridge.instances.get(id(instance), instance)
                if isinstance(instance, Instance) and instance.native is not None:
                    self._module_attribute(target.name, instance, target, writing=True)
                    value = self.evaluate(statement.value)
                    self.bridge.call(target, 'PyTorch attribute assignment failed',
                                     lambda: setattr(instance.native, target.name, value))
                    instance.properties[target.name] = True
                    return
                if not isinstance(instance, Instance):
                    self._error(target, 'Property assignment requires an instance')
                instance.properties[target.name] = self.evaluate(statement.value)
            else:
                collection = self.evaluate(target.collection)
                if type(collection) is not list:
                    self._error(target, 'Indexed assignment requires a list; arrays are read-only in Vexa')
                index = self._index(collection, self.evaluate(target.index), target)
                value = self.evaluate(statement.value)
                self._index(collection, index, target)
                collection[index] = value
        elif isinstance(statement, Ask):
            prompt = self.evaluate(statement.prompt)
            if type(prompt) is not str:
                self._error(statement, 'ask prompt must be a string')
            print(prompt, end='', file=self.output, flush=True)
            try:
                value = self.input_stream.readline()
            except (OSError, UnicodeError):
                self._error(statement, 'Unable to read terminal input')
            if value == '':
                self._error(statement, 'End of input while reading ask')
            value = value.rstrip('\r\n')
            try:
                owner = self.environment._owner(statement.name, statement.token)
            except VexaRuntimeError:
                owner = self.environment
            owner.values[statement.name] = value
        elif isinstance(statement, (Say, Print)):
            value = self.evaluate(statement.value)
            try:
                text = self.display(value)
            except ValueError:
                self._error(statement, 'Value is too large to display')
            print(text, end='' if isinstance(statement, Print) else '\n', file=self.output)
        elif isinstance(statement, ClassDeclaration):
            parent = None
            if statement.parent is not None:
                if statement.parent == statement.name:
                    self._error(statement, 'Inheritance cycle: a class cannot inherit from itself')
                if '.' in statement.parent:
                    if statement.parent != 'nn.Module':
                        self._error(statement, 'Unsupported Python base; only nn.Module is allowlisted')
                    self.bridge.require('pytorch', statement)
                    parent = ClassValue('nn.Module', {})
                else:
                    parent = self.environment.get(statement.parent, statement.token)
                if not isinstance(parent, ClassValue):
                    self._error(statement, 'Parent must be a class')
                seen = set()
                ancestor = parent
                while ancestor is not None:
                    if id(ancestor) in seen:
                        self._error(statement, 'Inheritance cycle')
                    seen.add(id(ancestor))
                    ancestor = ancestor.parent
            methods = {}
            klass = ClassValue(statement.name, methods, parent)
            for method in statement.methods:
                if not isinstance(method, Define):
                    self._error(statement, 'Class bodies may only contain method definitions')
                if method.name in methods:
                    self._error(method, f'Duplicate method {method.name!r}')
                if 'self' in method.parameters:
                    self._error(method, 'Method self is implicit; omit it from parameters')
                if klass.torch_base and method.name.startswith('_') and method.name != '__init__':
                    self._error(method, 'Unsupported private PyTorch method')
                methods[method.name] = Function(method, self.environment, klass)
            self.environment.define(statement.name, klass, statement.token)
        elif isinstance(statement, Define):
            if len(set(statement.parameters)) != len(statement.parameters):
                self._error(statement, 'Duplicate parameter in function declaration')
            self.environment.define(statement.name, Function(statement, self.environment),
                                    statement.token)
        elif isinstance(statement, Return):
            if self.function_depth == 0:
                self._error(statement, 'return is only valid inside a function')
            raise _ReturnSignal(self.evaluate(statement.value), statement)
        elif isinstance(statement, ExpressionStatement):
            self.evaluate(statement.expression)
        elif isinstance(statement, If):
            for condition, body in statement.branches:
                if self._boolean(self.evaluate(condition), condition):
                    self.execute_block(body)
                    break
            else:
                if statement.otherwise is not None:
                    self.execute_block(statement.otherwise)
        elif isinstance(statement, While):
            while self._boolean(self.evaluate(statement.condition), statement.condition):
                if not self._iteration(statement.body):
                    break
        elif isinstance(statement, Repeat):
            count = self.evaluate(statement.count)
            if type(count) is not int or count < 0:
                self._error(statement, 'Repeat count must be a nonnegative integer')
            for _ in range(count):
                if not self._iteration(statement.body):
                    break
        elif isinstance(statement, (Break, Continue)):
            if self.loop_depth == 0:
                self._error(statement, 'break/continue is only valid inside a loop')
            if isinstance(statement, Break):
                raise _BreakSignal()
            raise _ContinueSignal()
        elif isinstance(statement, CounterFor):
            start = self.evaluate(statement.start)
            end = self.evaluate(statement.end)
            step = self.evaluate(statement.step) if statement.step is not None else 1
            for value, node, label in ((start, statement.start, 'start'),
                                       (end, statement.end, 'end'),
                                       (step, statement.step or statement, 'step')):
                if type(value) is not int:
                    self._error(node, f'For loop {label} must be an integer')
            if step == 0:
                self._error(statement.step or statement, 'For loop step must not be zero')
            enclosing = self.environment
            self.environment = Environment(enclosing)
            self.environment.define(statement.name, start, statement.token)
            try:
                for counter in range(start, end, step):
                    self.environment.values[statement.name] = counter
                    if not self._iteration(statement.body):
                        break
            finally:
                self.environment = enclosing
        elif isinstance(statement, ForEach):
            values = self.evaluate(statement.iterable)
            torch = self.bridge.modules.get('pytorch')
            if type(values) is list:
                values = values.copy()
            elif torch is None or not isinstance(values, torch.utils.data.DataLoader):
                self._error(statement, 'for each requires a list or PyTorch DataLoader')
            for value in self._batch_iterator(values, statement):
                self.environment.values[statement.name] = value
                if not self._iteration(statement.body):
                    break
        else:
            self._error(statement, f'{type(statement).__name__} execution is reserved for a later chunk')

    def _module_attribute(self, name, instance, node, writing=False):
        t = self.bridge.require('pytorch', node)
        if name.startswith('_') or (writing and
                (name == 'training' or hasattr(t.nn.Module, name) or instance.klass.find_method(name) is not None)):
            self._error(node, 'Unsafe or reserved PyTorch attribute ' + repr(name))
        if writing and '_parameters' not in instance.native.__dict__:
            self._error(node, 'Call super().__init__() before assigning PyTorch attributes')

    def _invoke(self, callee, arguments, node):
        t = self.bridge.modules.get('pytorch')
        if isinstance(callee, ModuleInitializer):
            if arguments:
                self._error(node, 'nn.Module.__init__ expects 0 arguments')
            native = callee.instance.native
            if '_parameters' in native.__dict__:
                self._error(node, 'nn.Module parent is already initialized')
            t.nn.Module.__init__(native)
            return None
        if t is not None and isinstance(callee, t.nn.Module):
            return self.bridge.call(node, 'PyTorch forward failed (check shapes, types and devices)',
                                    lambda: callee(*arguments))
        if isinstance(callee, Builtin):
            return self._builtin(callee, arguments, node)
        if isinstance(callee, ClassValue):
            instance = Instance(callee, {})
            if callee.torch_base:
                runtime = self
                # Only forward is a Python callback. Vexa methods never replace
                # Module registration, device, hook, or serialization machinery.
                class VexaModule(t.nn.Module):
                    def forward(native, *args):
                        method = callee.find_method('forward')
                        if method is None:
                            runtime._error(node, 'PyTorch class requires a forward method')
                        return runtime.bridge.call(node, 'PyTorch forward failed',
                            lambda: runtime._invoke(BoundMethod(method, instance), list(args), node))
                instance.native = VexaModule.__new__(VexaModule)
                self.bridge.instances[id(instance.native)] = instance
            initializer = callee.find_method('__init__')
            if initializer is not None:
                self._invoke(BoundMethod(initializer, instance), arguments, node)
            elif arguments:
                self._error(node, f'Class {callee.name!r} expects 0 arguments, got {len(arguments)}')
            if instance.native is not None:
                if initializer is None:
                    t.nn.Module.__init__(instance.native)
                if '_parameters' not in instance.native.__dict__:
                    self._error(node, 'PyTorch constructor must call super().__init__()')
                return instance.native
            return instance
        instance = None
        if isinstance(callee, BoundMethod):
            instance = callee.instance
            callee = callee.function
        if not isinstance(callee, Function):
            self._error(node, 'Cannot call value: expected a function')
        declaration = callee.declaration
        if len(arguments) != len(declaration.parameters):
            self._error(node,
                        f'Function {declaration.name!r} expects '
                        f'{len(declaration.parameters)} arguments, got {len(arguments)}')
        environment = Environment(callee.closure)
        # Call-local context: ordinary nested/helper functions cannot borrow super.
        environment.method_context = (callee.defining_class, instance)
        if instance is not None:
            environment.define('self', instance.native if instance.native is not None else instance, declaration.token)
        for name, value in zip(declaration.parameters, arguments):
            environment.define(name, value, declaration.token)
        previous_environment = self.environment
        previous_loop_depth = self.loop_depth
        self.environment = environment
        self.loop_depth = 0
        self.function_depth += 1
        result = None
        return_node = declaration
        try:
            self.execute_block(declaration.body)
        except _ReturnSignal as returned:
            result = returned.value
            return_node = returned.node
        except RecursionError:
            self._error(node, 'Function call recursion limit exceeded')
        finally:
            self.environment = previous_environment
            self.loop_depth = previous_loop_depth
            self.function_depth -= 1
        if instance is not None and declaration.name == '__init__' and result is not None:
            self._error(return_node, "Constructor '__init__' must return null")
        if declaration.return_type is not None:
            expected = declaration.return_type
            torch = self.bridge.modules.get('pytorch')
            valid = (torch is not None and isinstance(result, torch.Tensor)) if expected == 'tensor' else type(result) is RETURN_TYPES[expected]
            if not valid:
                actual = next((name for name, kind in RETURN_TYPES.items()
                               if type(result) is kind), type(result).__name__)
                label = (f"Method {callee.defining_class.name + '.' + declaration.name!r}"
                         if callee.defining_class is not None
                         else f"Function {declaration.name!r}")
                self._error(return_node, f"{label} return type mismatch: "
                            f"expected {expected}, got {actual}")
        return result

    def evaluate(self, expression: Expr) -> Any:
        if isinstance(expression, TorchExpr):
            return self._torch(expression)
        if isinstance(expression, ArrayExpr):
            return self._array(expression)
        if isinstance(expression, TableExpr):
            return self._table(expression)
        if isinstance(expression, Literal):
            return expression.value
        if isinstance(expression, Super):
            klass, instance = self.environment.method_context
            if klass is None or instance is None:
                self._error(expression, 'super() is only valid inside an instance method')
            return SuperProxy(klass.parent, instance)
        if isinstance(expression, Name):
            return self.environment.get(expression.identifier, expression.token)
        if isinstance(expression, ListLiteral):
            return [self.evaluate(element) for element in expression.elements]
        if isinstance(expression, Property):
            instance = self.evaluate(expression.object)
            if isinstance(instance, Module):
                return self._module_member(instance, expression.name, expression)
            if isinstance(instance, SuperProxy):
                if instance.parent is not None and instance.parent.name == 'nn.Module':
                    if expression.name != '__init__':
                        self._error(expression, 'Only nn.Module super().__init__() is supported')
                    return ModuleInitializer(instance.instance)
                method = instance.parent.find_method(expression.name) if instance.parent else None
                if method is None:
                    self._error(expression, f'Undefined parent method {expression.name!r}')
                return BoundMethod(method, instance.instance)
            instance = self.bridge.instances.get(id(instance), instance)
            if isinstance(instance, Instance) and instance.native is not None:
                self._module_attribute(expression.name, instance, expression)
                if expression.name in instance.properties or expression.name == 'training':
                    return self.bridge.call(expression, 'PyTorch attribute access failed',
                                            lambda: getattr(instance.native, expression.name))
                method = instance.klass.find_method(expression.name)
                if method is not None:
                    return BoundMethod(method, instance)
                self._error(expression, 'Unsupported PyTorch attribute ' + repr(expression.name))
            if not isinstance(instance, Instance):
                self._error(expression, 'Property access requires an instance')
            if expression.name in instance.properties:
                return instance.properties[expression.name]
            method = instance.klass.find_method(expression.name)
            if method is not None:
                return BoundMethod(method, instance)
            self._error(expression, f'Undefined property {expression.name!r}')
        if isinstance(expression, Index):
            collection = self.evaluate(expression.collection)
            index = self._index(collection, self.evaluate(expression.index), expression)
            value = collection[index]
            np = self.bridge.modules.get('numpy')
            return value.item() if np is not None and isinstance(value, np.generic) else value
        if isinstance(expression, Call):
            callee = self.evaluate(expression.callee)
            arguments = [self.evaluate(argument) for argument in expression.arguments]
            return self._invoke(callee, arguments, expression)
        if isinstance(expression, Unary):
            value = self.evaluate(expression.operand)
            if expression.operator == 'not':
                return not self._boolean(value, expression)
            if not self._number(value):
                self._error(expression, 'Unary arithmetic requires a number')
            return +value if expression.operator == '+' else -value
        if not isinstance(expression, Binary):
            self._error(expression, f'{type(expression).__name__} evaluation is reserved for a later chunk')
        left = self.evaluate(expression.left)
        op = expression.operator
        if op in ('and', 'or'):
            left = self._boolean(left, expression)
            if (op == 'and' and not left) or (op == 'or' and left):
                return left
            return self._boolean(self.evaluate(expression.right), expression)
        right = self.evaluate(expression.right)
        numeric = self._number(left) and self._number(right)
        if op in ('is equal to', 'is not equal to'):
            equal = self._equal(left, right)
            return equal if op == 'is equal to' else not equal
        if op.startswith('is '):
            if not (numeric or type(left) is type(right) is str):
                self._error(expression, 'Ordering requires two numbers or two strings')
            if op == 'is greater than':
                return left > right
            if op == 'is less than':
                return left < right
            if op == 'is greater than or equal to':
                return left >= right
            if op == 'is less than or equal to':
                return left <= right
            self._error(expression, f'Unknown comparison {op!r}')
        if op == '+' and type(left) is type(right) and type(left) in (str, list):
            return left + right
        if not numeric:
            self._error(expression, 'Arithmetic requires numbers; + also accepts two strings or two lists')
        try:
            if op == '+':
                result = left + right
            elif op == '-':
                result = left - right
            elif op == '*':
                result = left * right
            elif op == '/':
                result = left / right
            elif op == '%':
                result = left % right
            elif op == '^':
                result = left ** right
            else:
                self._error(expression, f'Unknown operator {op!r}')
        except ZeroDivisionError:
            self._error(expression, 'Division by zero')
        except (OverflowError, ValueError):
            self._error(expression, 'Numeric result is out of range')
        if type(result) is complex:
            self._error(expression, 'Complex numeric results are not supported')
        if type(result) is float and not math.isfinite(result):
            self._error(expression, 'Numeric result is not finite')
        return result


def check_syntax(source: str) -> list[dict]:
    """Return the first syntax diagnostic; never evaluate or execute the AST."""
    try:
        tokens = Lexer(source).tokenize()
        Parser(tokens).parse()
        return []
    except VexaError as error:
        message = str(error).split(': ', 1)[1]
        # Preserve the real parser's location and add actionable English guidance.
        if 'tokens' in locals():
            token = next((t for t in tokens if (t.line, t.column) ==
                          (error.line, error.column)), None)
            if token and token.kind in {'GREATER', 'LESS', 'EQUAL'}:
                message += "; comparisons must start with 'is'"
            if token and token.kind == 'BE' and "Expected 'to'" in message:
                message += "; use 'set name to value.'"
        return [{'line': error.line, 'column': error.column, 'message': message,
                 'incomplete': 'end of input' in message}]
    except (RecursionError, ValueError, OverflowError):
        return [{'line': 1, 'column': 1, 'message': 'Source exceeds syntax checker limits.',
                 'incomplete': False}]


def run(source: str, *, environment: Environment | None = None, output=None, input_stream=None,
        source_path: str | Path | None = None) -> Interpreter:
    """Parse a complete source file before executing any statements."""
    path = Path(source_path).resolve() if source_path is not None else None
    program = Parser(Lexer(source, path).tokenize()).parse()
    interpreter = Interpreter(environment, output, input_stream)
    interpreter.execute(program)
    return interpreter


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Execute a Vexa source file.')
    parser.add_argument('--language-metadata', action='store_true',
                        help='print editor metadata as JSON without running a program')
    parser.add_argument('file', nargs='?', help='path to a .vexa file')
    arguments = parser.parse_args(argv)
    if arguments.language_metadata:
        print(json.dumps(language_metadata()))
        return 0
    if arguments.file is None:
        parser.error('the following arguments are required: file')
    path = Path(arguments.file)
    try:
        source = path.read_text(encoding='utf-8')
        run(source, source_path=path)
    except VexaError as error:
        print(f'{error.source_path or path}:{error}', file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as error:
        print(f'{path}: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
