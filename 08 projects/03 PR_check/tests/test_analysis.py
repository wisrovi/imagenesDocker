import pytest
from unittest.mock import patch
from pr_analyzer.analysis import run_pylint_on_content
import textwrap

def test_run_pylint_on_content_success():
    """
    Tests the successful execution of the Pylint analysis.
    """
    # A simple python code with a known pylint score
    code_content = textwrap.dedent('''\
        '''A very simple module.'''

        def func():
            '''A very simple function.'''
            pass
    ''')
    
    score = run_pylint_on_content(code_content)
    
    assert isinstance(score, float)
    assert score > 0.0

def test_run_pylint_on_content_fail():
    """
    Tests the graceful failure of the Pylint analysis.
    """
    # A python code with syntax error
    code_content = "a = 1\nb = \n"
    
    score = run_pylint_on_content(code_content)
    
    assert score == 0.0