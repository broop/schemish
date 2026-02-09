# schemish

A modified version of Peter Norvig's Lispy (c) 2010-16; See http://norvig.com/lispy.html 

schemish is tailored to run pretty much only the code from The Little Schemer.

This version has the following features not found in the original `lis.py`:
- uses generators to avoid Python's recursion limit (aka trampolining).
- handles comments
- has a nicer repl (imo)
- can load files (see  example below)
- tokenization is more robust
- supports multiple expressions in a body.

I have left everything else as is, and have not changed names of variables or functions unless necessary.

The book code `ls.scm` can be found at https://github.com/broop/little-schemer/ - It was originally developed in Racket.

Note: I've left most of the original comments, except where they became irrelevant or are no longer correct due to modification.

### Usage:
```
python3 schemish2.py ls.scm
>>> (! 10)
3628800
```
