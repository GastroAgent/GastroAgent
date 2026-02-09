from functools import lru_cache
import re
import json
import signal
from collections import Counter
from sympy import sympify, simplify

from typing import Dict, List

from tqdm import tqdm

class TimeoutException(Exception):
    pass

def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s: str):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def parse_math_boxed(s: str):
    if not s:
        return "N/A"
    s = last_boxed_only_string(s)
    s = remove_boxed(s)
    return s

import regex
def parse_my_boxed_plus(answer: str):
    pattern = r'\\boxed\{((?:[^{}]++|(?R))*)\}'
    results = regex.findall(pattern, answer)
    if len(results) == 0:
        pattern = r'\*\*\[*(.*?)\]*\*\*(?!:)'
        results = re.findall(pattern, answer)
        if len(results) == 0 or ':' in results[-1]:
            return None
    return results[-1] if len(results) > 0 else None

def parse_my_box_plus(answer: str):
    pattern = r'\\box\{((?:[^{}]++|(?R))*)\}'
    results = regex.findall(pattern, answer)
    if len(results) == 0:
        pattern = r'\*\*\[*(.*?)\]*\*\*(?!:)'
        results = re.findall(pattern, answer)
        if len(results) == 0 or ':' in results[-1]:
            return None
    return results[-1] if len(results) > 0 else None

def parse_my_boxed(answer: str):
    pattern = r'\\boxed{(.*)}[\.\n\s]*'
    results = re.findall(pattern, answer)
    if len(results) == 0:
        pattern = r'\*\*\[*(.*?)\]*\*\*(?!:)'
        results = re.findall(pattern, answer)
        if len(results) == 0 or ':' in results[-1]:
            return None
    return results[-1] if len(results) > 0 else None

def parse_my_box(answer: str):
    pattern = r'\\box{(.*)}[\.\n\s]*'
    results = re.findall(pattern, answer)
    if len(results) == 0:
        pattern = r'\*\*\[*(.*?)\]*\*\*(?!:)'
        results = re.findall(pattern, answer)
        if len(results) == 0 or ':' in results[-1]:
            return None
    return results[-1] if len(results) > 0 else None

def parse_data_(data: Dict[str, str], dummy_answer: List[str], key='best_answer', output_key='only_answer'):
    answer = data[key]
    only_answer = parse_my_boxed(answer)
    if only_answer is None:
        only_answer = extract_boxed_answer(answer)
    if only_answer is None:
        try:
            only_answer = re.findall(r"the (?:final )?answer is (.*?)(?:\.|$)",  answer.lower())[-1]
            if only_answer in dummy_answer:
                raise ValueError
        except:
            answer = data['feedback']
            only_answer = parse_my_boxed(answer)
            if only_answer is None:
                only_answer = parse_math_boxed(answer)
            if only_answer is None:
                only_answer = extract_boxed_answer(answer)
            if only_answer is None:
                try:
                    only_answer = only_answer = re.findall(r"the (?:final )?answer is (.*?)(?:\.|$)",  answer.lower())[-1]
                    if only_answer not in dummy_answer:
                        data['feedback_answer'] = data['feedback']
                except:
                    only_answer = 'N/A'
            else:
                data['feedback_answer'] = data['feedback']
                
    data[output_key] = only_answer

def parse_answer_old(reasoning: str):
    only_answer = parse_my_boxed(reasoning)
    if only_answer is None:
        only_answer = parse_my_box(reasoning)
    if only_answer is None:
        only_answer = extract_boxed_answer(reasoning)
    if only_answer is None:
        pattern = r'Final Answer is (.*)?'
        results = re.findall(pattern, reasoning)
        if len(results) == 0:
            only_answer = None
        else:
            only_answer = results[-1]
    return only_answer.lower().split('$')[0].split('\\[')[0] if only_answer is not None else ''

def parse_answer(reasoning: str):
    only_answer = parse_my_boxed_plus(reasoning)
    if only_answer is None:
        only_answer = parse_my_box_plus(reasoning)
    if only_answer is None:
        only_answer = parse_my_boxed(reasoning)
    if only_answer is None:
        only_answer = parse_my_box(reasoning)
    if only_answer is None:
        only_answer = extract_boxed_answer(reasoning)
    if only_answer is None:
        pattern = r'Final Answer is (.*)?'
        results = re.findall(pattern, reasoning)
        if len(results) == 0:
            only_answer = None
        else:
            only_answer = results[-1]
    return only_answer.lower().split('\\[')[0] if only_answer is not None else ''

def is_complex_number(complex_str):
    pattern = r'[+-]*(\d*)i[+-]*(\d*)'
    match1 = re.match(pattern, complex_str.replace(' ', ''))
    pattern = r'[+-]*(\d*)[+-]*(\d*)i'
    match2 = re.match(pattern, complex_str.replace(' ', ''))
    return bool(match1) or bool(match2)

def parse_complex_number(complex_str):
    pattern = r'([+-]*\d*)i[+-]*'
    match = re.findall(pattern, complex_str.replace(' ', ''))
    if not len(match) == 0 and not len(match[0]) == 0:
        imag = match[0]
    else:
        imag = 0
    pattern =  r'[+-]*\d+\b'
    match = re.findall(pattern, complex_str.replace(' ', ''))
    if not len(match) == 0 and not len(match[0]) == 0:
        real = match[0]
    else:
        real = 0
    try:
        return float(real), float(imag)
    except ValueError:
        return float(0), float(0)
    
def compare_complex_numbers(str1, str2):
    if 'i' not in str1 and 'i' not in str2 and 'j' not in str1 and 'j' not in str2 and 'k' not in str1 and 'k' not in str2:
        return False
    
    if  is_complex_number(str1) and is_complex_number(str2):
        real1, imag1 = parse_complex_number(str1)
        real2, imag2 = parse_complex_number(str2)
        result = (real1, imag1) == (real2, imag2)
        if result: return result
    
    if  is_complex_number(str1.replace('j', 'i')) and is_complex_number(str2.replace('j', 'i')):
        real1, imag1 = parse_complex_number(str1.replace('j', 'i'))
        real2, imag2 = parse_complex_number(str2.replace('j', 'i'))
        result = (real1, imag1) == (real2, imag2)
        
    if is_complex_number(str1.replace('k', 'i')) and is_complex_number(str2.replace('k', 'i')):
        real1, imag1 = parse_complex_number(str1.replace('k', 'i'))
        real2, imag2 = parse_complex_number(str2.replace('k', 'i'))
        result = (real1, imag1) == (real2, imag2)
        if result:
            return result
    return False

def evaluate_math(results: List[Dict[str, str]], pred_key='only_answer', gt_key='GT_answer'):
    num_correct = 0
    for result in tqdm(results, desc='eval result:'):
        if is_math_correct(result[pred_key], result[gt_key]):
            num_correct += 1
    acc = round((num_correct / len(results)), 4)
    return acc

def check_text(str1, str2):
    compare1 = re.findall(r'\\text{(.*?)}', str1)
    if not len(compare1) == 0:
        compare1 = compare1[0].lower()
    else:
        compare1 = str1.lower()
    compare2 = re.findall(r'\\text{(.*?)}', str2)
    if not len(compare2) == 0:
        compare2 = compare2[0].lower()
    else:
        compare2 = str2.lower()
    return compare1 == compare2


def check_remove_dict(str1, str2):
    str1 = str1.replace(' ', '')
    str2 = str2.replace(' ', '')
    return str1.replace('{', '').replace('}', '').replace('units', '').replace('square', '') == str2.replace('{', '').replace('}', '').replace('units', '').replace('square', '')

def check_final(str1, str2):
    reps = ['cm', r'\\%']
    for rep in reps:
        str1 = str1.replace(rep, '')
        str2 = str2.replace(rep, '')
    return str1 == str2
def check_remove_left(str1, str2):
    return str1.replace('\left', '').replace(r'\right', '').replace(' ','') == str2.replace('\left', '').replace(r'\right', '').replace(' ','')

def check_star(str1, str2):
    pred = re.findall(r'\*\*(.*?)°*\*\*', str1)
    if len(pred) == 0:
        pred = str1
    else:
        pred = pred[0]
    gt = re.findall(r'\*\*(.*?)°*\*\*', str2)
    if len(gt) == 0:
        gt = str2
    else:
        gt = gt[0]
    return pred.replace(' ', '') == gt.replace(' ', '')

def check_dollar(str1, str2):
    pred = re.findall(r'\$*(.*?)°*\$', str1)
    if len(pred) == 0:
        pred = str1
    else:
        pred = pred[0]
    gt = re.findall(r'\$*(.*?)°*\$', str2)
    if len(gt) == 0:
        gt = str2
    else:
        gt = gt[0]
    return pred.replace(' ', '') == gt.replace(' ', '')


@lru_cache(1024)
def extract_label(text: str,type='') -> str:
    if type != 'digit':
        if '####' in text:
            text = text.split('####')[-1]
        elif 'The answer is' in text:
            text = text.split('The answer is')[-1]
            if '####' in text:
                text = text.split('####')[-1]
        if 'box' in text:
            return extract_boxed_answer(text)
        else:
            return text
    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',','')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',','')
    pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
    numbers = pattern.findall(text)
    if not numbers:
        return None
    if '\n####' in text or 'The answer is' in text:
        return numbers[0]
    else :
        return numbers[-1]

@lru_cache(1024)
def check(gt,ans):
    gt_label = extract_label(gt)
    
    type_ = 'formula'
    ans_label = extract_label(ans,type_)
    if ans_label:
        ans_label = ans_label.replace('$','')
    return is_equiv(gt_label,ans_label)

def is_math_correct(pred, gts):
    if pred is None:
        return False
    if isinstance(gts, str):
        gts = [gts]
    pred2 = pred
    pred = re.findall(r'\$*(.*?)\$*', pred)[0] if '$' in pred else pred
    pred = pred if not len(pred) == 0 else pred2
    pred = pred.split('=')[-1].split('^\\')[0].split('^{\\')[0].split('\\$')[-1].split('\\%')[-1].split('\\(')[-1].split('\\)')[0].split('\\,')[0].split('cm²')[0].strip()
    gts = [ gt.split('=')[-1].split('^\\')[0].split('^{\\')[0].split('\\$')[-1].split('\\mbox')[0].strip() for gt in gts]
    if pred.replace(' ','').replace('.','').lower() in [gt.lower().replace(' ','') if isinstance(gt, str) else gt for gt in gts if gt is not None]:
        return True
    for gt in gts:
        if math_check1(pred2, gt):
            return True
        if math_check2(pred2, gt):
            return True
        if check(pred2, gt):
            return True
        if compare_complex_numbers(pred, gt):
            return True
        if check_text(pred, gt):
            return True
        if check_remove_dict(pred, gt):
            return True
        if check_remove_left(pred, gt):
            return True
        if check_star(pred, gt):
            return True
        if check_dollar(pred, gt):
            return True
        if check(pred, gt):
            return True
    return False

def math_check1(pred, gt):

    signal.signal(signal.SIGALRM, TimeoutException)
    signal.alarm(5)  # Set the alarm for timeout_duration seconds
    try:
        if is_frac_equiv(pred, gt):
            return True
        if is_string_equiv(pred, gt):
            return True
        numeric_gt_value = get_fraction_value(gt)
        numeric_pred_value = get_fraction_value(pred)
        if within_eps(pred, numeric_gt_value):
            return True
        if within_eps(numeric_pred_value, numeric_gt_value):
            return True            
        numeric_gt_value = get_latex_value(gt)
        numeric_pred_value = get_latex_value(pred)
        if within_eps(pred, numeric_gt_value):
            return True  
        if within_eps(numeric_pred_value, numeric_gt_value):
            return True  
    except TimeoutException:
        return False
    finally:
        signal.alarm(0)
    return False
def math_check2(pred, gt):
    if math_check1(_remove_right_units(pred), _remove_right_units(gt)):
        return True
    return False

def is_frac_equiv(expr1, expr2):
    """
    Determines whether two mathematical expressions, possibly containing different fraction notations,
    are equivalent.

    :param expr1: A string representing the first mathematical expression.
    :param expr2: A string representing the second mathematical expression.
    :return: True if the expressions are equivalent, False otherwise.
    """
    try:
        # Normalize fraction notations
        expr1_sympy = normalize_fraction_notation(expr1)
        expr2_sympy = normalize_fraction_notation(expr2)

        # Convert the string expressions into sympy expressions
        sympy_expr1 = sympify(expr1_sympy)
        sympy_expr2 = sympify(expr2_sympy)

        # Simplify both expressions and check for equality
        return simplify(sympy_expr1 - sympy_expr2) == 0
    except Exception:
        
        return False

def normalize_fraction_notation(expr):
    """
    Normalizes different fraction notations (\\frac, \frac, rac) into a consistent format.

    :param expr: A string containing the expression with fraction notations.
    :return: A string with the normalized fraction format.
    """
    # Regular expression to find different fraction notations
    frac_pattern = r"(\\\\d*t*frac|\\d*t*frac|rac)\{([^}]+)\}\{([^}]+)\}"

    # Function to replace the fraction notations with (numerator)/(denominator)
    def replace_frac(match):
        _, num, den = match.groups()
        return f"({num})/({den})"

    return re.sub(frac_pattern, replace_frac, expr)

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    string = str(string)
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_string_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def get_fraction_value(expr):
    try:
        expr = expr.replace('\pi', '3.1416')
        frac_pattern = r"(\\\\d*t*frac|\\d*t*frac|rac)\{([^}]+)\}\{([^}]+)\}"
        def replace_frac(match):
            _, num, den = match.groups()
            return f"{num}/{den}"
        return float(eval(re.sub(frac_pattern, replace_frac, expr)))
    except Exception:
        return expr

def floatify(num):
    try:
        num = float(num)
        return num
    except Exception as e:
        return None

def within_eps(pred, gt):
    pred = floatify(pred)
    gt = floatify(gt)
    if pred is None or gt is None:
        return False
    eps = abs(gt-pred)
    if eps < 0.01:
        return True
    else:
        return False

def get_latex_value(expr):
    try:
        if '\\sqrt' in expr and expr.split("\\sqrt")[0]:
            multi_str = expr.split("\\sqrt")[0]
            multiplier = float(multi_str.strip())
            multiplicand = "\\" + expr.split(multi_str+"\\")[-1]
            value = multiplier * float(latex2sympy(multiplicand).evalf())
        else:    
            value = float(latex2sympy(expr).evalf())
    except:
        return expr
    return value

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

def extract_matching_bracket(target_str: str):
    if not target_str:
        return target_str
    current_nest_level = 1
    for i, ch in enumerate(target_str):
        if ch == '{':
            current_nest_level += 1
        elif ch == '}':
            current_nest_level -= 1
        if current_nest_level == 0:
            break
    return target_str[:i]

def clean(target_str: str):
    opt = target_str.strip().replace('{{', '{').replace('}}', '}')
    if not opt:
        return opt
    if opt[-1] == '.' or opt[-1] == '。':
        return opt[:-1]
    return opt

def extract_answer(pred: str, extract_last_num=False):
    if pred.find('The final answer is ') >= 0:
        x = pred[pred.find('The final answer is ') +
                 len('The final answer is '):]
        x = x[1:x.find('$.')]
            # print(x)
        return clean(x)
    if pred.find('\n\nQuestion:') >= 0:
        pred = pred.split('\n\nQuestion:')[0]
        if pred.find('The answer is'):
            pred = pred[pred.find('The answer is') + len('The answer is'):]
            return clean(pred)
    if pred.find('# Answer') >= 0:
        return clean(pred[pred.find('# Answer') + len('# Answer'):])
    if pred.find('The answer is:') >= 0:
        return clean(pred[pred.find('The answer is:') +
                              len('The answer is:'):])
    if pred.find('####') >= 0:
        return clean(pred[pred.find('####') + 4:])
    left = '\\boxed{'
    if pred.find(left) >= 0:
        pred = pred[pred.find(left) + len(left):]
        return clean(extract_matching_bracket(pred))

    if extract_last_num:
        nums = []
        opt = ''

        def contain_digit(opt):
            for ch in opt:
                if ch.isdigit():
                    return True
            return False

        for ch in pred:
            if ch.isdigit() or ch in ' ,.':
                opt = opt + ch
            else:
                if contain_digit(opt):
                    nums.append(opt)
                opt = ''
        if contain_digit(opt):
            return cls.clean(opt)
        if nums:
            return cls.clean(nums[-1])
        
    return None
    
def fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set)
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string

def strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()  # noqa: W605

    # a_b == a, a_{b} == a_b for bit conversion
    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    # 10800 == 10,800; we only deal with single number
    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return False
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
