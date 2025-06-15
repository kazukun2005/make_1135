from django.shortcuts import render

from django.shortcuts import render
import numpy as np
import random
import numpy as np
from collections import defaultdict
import ast
import re
memo = {}
def solve_recursively(nums):
    if nums in memo: return memo[nums]
    if len(nums) == 1: return [(nums[0], str(nums[0]))]
    results = []
    for i in range(1, len(nums)):
        left_part, right_part = nums[:i], nums[i:]
        left_results, right_results = solve_recursively(left_part), solve_recursively(right_part)
        for val_l, expr_l in left_results:
            for val_r, expr_r in right_results:
                results.append((val_l + val_r, f"({expr_l}+{expr_r})"))
                results.append((val_l - val_r, f"({expr_l}-{expr_r})"))
                results.append((val_l * val_r, f"({expr_l}*{expr_r})"))
                if val_r != 0 and val_l % val_r == 0:
                    results.append((val_l // val_r, f"({expr_l}/{expr_r})"))
    memo[nums] = results
    return results

def get_concatenated_partitions(numbers):
    if not numbers: yield []; return
    for i in range(1, len(numbers) + 1):
        current_num = int("".join(map(str, numbers[:i])))
        for rest_partition in get_concatenated_partitions(numbers[i:]):
            yield [current_num] + rest_partition

# --- ここからが変更部分 ---

def _negate_ast_node(node):
    """ASTノードの符号を、順序を維持したまま代数的に反転させる"""
    if isinstance(node, ast.Constant): return ast.Constant(value=-node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub): return node.operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # -(A + B) -> -A - B
        return ast.BinOp(left=_negate_ast_node(node.left), op=ast.Sub(), right=node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        # -(A - B) -> -A + B (以前は B - A だった)
        # この変更により、式の順序が維持される
        return ast.BinOp(left=_negate_ast_node(node.left), op=ast.Add(), right=node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return ast.BinOp(left=_negate_ast_node(node.left), op=ast.Mult(), right=node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return ast.BinOp(left=_negate_ast_node(node.left), op=ast.Div(), right=node.right)
    return ast.UnaryOp(op=ast.USub(), operand=node)

# --- 既存の式簡略化関数 (変更なし) ---
def get_negated_expression(expr_str):
    try:
        tree = ast.parse(expr_str, mode='eval')
        negated_node = _negate_ast_node(tree.body)
        return _format_node(negated_node)
    except Exception:
        return f"-({expr_str})"

PRECEDENCE = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2}
OPERATOR_MAP = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}

def _format_node(node, parent_precedence=0):
    if isinstance(node, ast.Constant): return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        current_precedence = 100
        result = f"-{_format_node(node.operand, current_precedence)}"
        if current_precedence < parent_precedence: return f"({result})"
        return result
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        current_precedence = PRECEDENCE.get(op_type, 100)
        left_str = _format_node(node.left, current_precedence)
        right_parent_precedence = current_precedence + 1 if op_type in (ast.Sub, ast.Div) else current_precedence
        right_str = _format_node(node.right, right_parent_precedence)
        op_symbol = OPERATOR_MAP[op_type]
        result = f"{left_str}{op_symbol}{right_str}"
        if current_precedence < parent_precedence: return f"({result})"
        else: return result
    return ""

class MinusDistributor(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Sub) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Add):
            new_node = ast.BinOp(left=ast.BinOp(left=node.left, op=ast.Sub(), right=node.right.left), op=ast.Sub(), right=node.right.right)
            return self.visit(new_node)
        if isinstance(node.op, ast.Sub) and isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Sub):
            new_node = ast.BinOp(left=ast.BinOp(left=node.left, op=ast.Sub(), right=node.right.left), op=ast.Add(), right=node.right.right)
            return self.visit(new_node)
        return node

def simplify_expression(expr_str):
    if not isinstance(expr_str, str): return expr_str
    try:
        tree = ast.parse(expr_str, mode='eval')
        transformer = MinusDistributor()
        transformed_tree = transformer.visit(tree)
        return _format_node(transformed_tree.body)
    except Exception:
        return expr_str

def remove_unnecessary_parentheses(expression: str) -> str:
    """
    Parses a mathematical expression string, removes redundant parentheses,
    and returns the simplified string.

    This function preserves the mathematical meaning and order of operations
    while making the expression more readable.

    Args:
        expression: The mathematical expression as a string. 
                    It can use '×' for multiplication and '^' for exponentiation.

    Returns:
        The expression string with unnecessary parentheses removed.
        Returns an error message if the expression is invalid.
    """
    
    # ヘルパー関数：ASTノードから演算子の情報（記号と優先順位）を取得
    def _get_op_info(op_node):
        op_type = type(op_node)
        # 標準的な演算子の優先順位
        if op_type is ast.Add: return ('+', 1)
        if op_type is ast.Sub: return ('-', 1)
        if op_type is ast.Mult: return ('×', 2)
        if op_type is ast.Div: return ('/', 2)
        if op_type is ast.Pow: return ('^', 3)  # べき乗
        if op_type is ast.USub: return ('-', 4) # 単項マイナス
        return ('?', 99) # 不明な演算子の場合

    # 再帰ヘルパー関数：ASTを走査して文字列を構築
    def _format_node(node):
        # 基底ケース：数値や定数
        if isinstance(node, ast.Constant): # Python 3.8以降
            return str(node.value), 100 # リテラルは最高の優先順位を持つ
        if isinstance(node, ast.Num): # 古いPythonバージョン用
            return str(node.n), 100

        # 単項演算（例: -5）
        if isinstance(node, ast.UnaryOp):
            op_str, op_prec = _get_op_info(node.op)
            operand_str, operand_prec = _format_node(node.operand)

            # オペランド（操作される数）の優先順位が低い場合はかっこで囲む
            # 例: -(a + b) は "a + b" をかっこで囲む必要がある
            if operand_prec < op_prec:
                operand_str = f"({operand_str})"
            
            return f"{op_str}{operand_str}", op_prec

        # 二項演算（例: a + b）
        if isinstance(node, ast.BinOp):
            op_str, op_prec = _get_op_info(node.op)
            
            left_str, left_prec = _format_node(node.left)
            right_str, right_prec = _format_node(node.right)

            # 左オペランドの演算子の優先順位が低い場合はかっこを追加
            # 例: (a + b) * c の場合、 '+' は '*' より優先順位が低い
            if left_prec < op_prec:
                left_str = f"({left_str})"
            
            # 右オペランドにかっこを追加するケース
            # 1. 優先順位が低い場合 (例: c * (a + b))
            if right_prec < op_prec:
                right_str = f"({right_str})"
            # 2. 優先順位が同じで、親の演算子が左結合性でない場合
            #    例: a - (b - c) や a / (b / c)
            elif right_prec == op_prec:
                if isinstance(node.op, (ast.Sub, ast.Div)):
                     right_str = f"({right_str})"

            return f"{left_str}{op_str}{right_str}", op_prec

        raise TypeError(f"Unsupported node type: {type(node)}")

    try:
        # PythonのASTパーサー用に文字列を前処理
        py_expr = expression.replace('×', '*').replace('^', '**')
        
        # 式を抽象構文木（AST）にパース
        tree = ast.parse(py_expr, mode='eval')
        
        # ASTのトップノードから再帰的に文字列を構築
        simplified_expr, _ = _format_node(tree.body)
        
        return simplified_expr
    except (SyntaxError, TypeError, ValueError, RecursionError) as e:
        return f"式を処理中にエラーが発生しました (Error processing expression): {e}"
# --- メイン処理関数 (一部変更) ---
def generate_and_evaluate_expressions(initial_num_values):
    if not isinstance(initial_num_values, list) or len(initial_num_values) < 2:
        print("Error: Please provide a list with at least two numbers.")
        return defaultdict(list)
    
    memo.clear()
    interim_results = defaultdict(set)
    for partition in get_concatenated_partitions(initial_num_values):
        for value, expr in solve_recursively(tuple(partition)):
            if np.isnan(value) or np.isinf(value): continue
            value = int(value) if float(value).is_integer() else value
            interim_results[value].add(expr)

    final_results = defaultdict(set)
    for value, expressions in interim_results.items():
        target_key = abs(value)
        for expr in expressions:
            if value >= 0:
                final_results[target_key].add(expr)
            else:
                positive_expr = get_negated_expression(expr)
                final_results[target_key].add(positive_expr)

    formatted_results = defaultdict(list)
    for key in sorted(final_results.keys()):
        # 全ての式を不要な括弧で囲むのをやめ、簡略化のみ行う
        simplified = {f"({simplify_expression(e)})" for e in final_results[key]}
        formatted_results[key] = sorted(list(simplified), key=lambda s: (len(s), s))
            
    return formatted_results
def fast_sieve(limit):
    sieve = np.ones(limit + 1, dtype=bool)  # True = 素数候補
    sieve[:2] = False  # 0と1は素数ではない

    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i : limit+1 : i] = False  # iの倍数をFalseにする

    return np.flatnonzero(sieve)  # Trueのインデックス = 素数
def ys(num, sieve):
    lis = []
    for i in sieve:
        while num % i == 0:
            lis.append(int(i))
            num //= i
    return lis, int(num)

def add_mult(lis, num=1):
    if num == 1:
        retu = []
    else:
        retu = [str(num)]
    lis_l = len(lis)
    for i, j in enumerate(lis):
        retu.append(str(j))
        if lis_l-1 != i:
            retu.append("*")
    return retu
def check(ele):
    if ele == "ok" or ele == 1 or ele == "*" or ele == "+" or ele == "-" or ele =="(" or ele == ")":
        return True
    return False


def num_last(num,dic_copy):
    return_list = []
    if int(num) in num_1135_set:
        return dic_1135[int(num)][random.randint(0, len(dic_1135[int(num)]) - 1)]
    now_last = dic_copy[str(num)]
    for i,j in enumerate(now_last):
        if j == "ok" or j == 1 or j == "*" or j == "+" or j == "-" or j == ")" or j == "(" or j in num_1135_set or str(j)[0] == "^":
            return_list.append(j)
            continue
        else:
            next_last = num_last(j, dic_copy)
            return_list.append(next_last)
    return "".join(map(str, return_list))
def return_f(n_n, number):
    next = n_n if number >= 0 else f"-{n_n}"
    return next

def make_pow(lis):
    return_lis = []
    l = 0
    used = set()
    l1_len = len(lis)
    for l in range(len(lis)):
        if lis[l] == "*":
            pass
        elif lis[l] in used:
            pass
        else:
            now_c = lis.count(lis[l])
            if l !=0:return_lis.append("*")
            if now_c>1:
                return_lis.append(lis[l])
                used.add(lis[l])
                if num not in dic:
                    make_dic(now_c)
                moji = num_last(now_c, dic)
                return_lis.append(f"^{moji}")
            else:
                return_lis.append(lis[l])
    return return_lis
def make_dic(i):
    now = [i]
    while now:
        n = now.pop()
        if n in dic or n in num_1135_set or n == 1 or n =="*" or n == "+" or n == "-" or str(n)[0] == "^":
            continue
        n = int(n)
        l1, l2 = ys(n, num_1135)

        if l2 == n:
            dic[str(n)] = ["(",str(l2//2),"+", str(l2-l2//2),")"]
            now.extend([l2//2, l2-l2//2])
        else:
            addl = l1 + [str(l2)] if l2 != 1 else l1
            addl = add_mult(addl)
            if n in num_1135_set:dic[str(n)] = "ok"
            else:dic[str(n)] = make_pow(addl)
            now.extend(addl)
def sub_main(inp):
    global aa
    num = int(inp)
    if num in dic_1135:
        return dic_1135[num][random.randint(0, len(dic_1135[num]) - 1)]

    hh, m = ys(num, num_1135)
    
    kk, k2 = ys(m, aa)
    acc_lis = ["+", "-", "*"]

    """
    追加したい要素
    +-が並列している
    1135dicを直す
    ×にする
    """

    ans_lis = add_mult(hh)

    if k2 != 1:
        ret = [k2]
        ans_lis = [str(k2),"*"] + ans_lis
    else:
        ret = []

    ans_l = len(ans_lis)

    for i, j in enumerate(kk):
        if j not in num_1135_set:
            ret.append(j)
        if i == 0 and ans_l >= 1:
            ans_lis.append(acc_lis[2])
        ans_lis.append(str(j))
        if i != len(kk) - 1:
            ans_lis.append(acc_lis[2])

    last = list(set(ret))

    for i in last:
        now = [i]
        while now:
            n = now.pop()
            if n in dic or n in num_1135_set or n == 1 or n =="*" or n == "+" or n == "-" or str(n)[0] == "^":
                continue
            n = int(n)
            l1, l2 = ys(n, num_1135)

            if l2 == n:
                l_num = random.randint(1, l2)
                r_num = l2 - l_num
                dic[str(n)] = ["(",str(l_num),"+", str(r_num),")"]
                now.extend([l_num, r_num])
            else:
                addl = l1 + [str(l2)] if l2 != 1 else l1
                addl = add_mult(addl)
                if n in num_1135_set:dic[str(n)] = "ok"
                else:dic[str(n)] = make_pow(addl)
                now.extend(addl)

    if ans_lis[-1] == "*":
        ans_lis.pop()
    ans_dic = {}
    for i in last:
        ans = num_last(i,dic)
        ans_dic[str(i)] = ans
    ans_list = []

    for i in make_pow(ans_lis):
        if i in num_1135_str:
            ans_list.append(dic_1135[int(i)][random.randint(0, len(dic_1135[int(i)]) - 1)])
        elif i in ans_dic:
            ans_list.append(ans_dic[i])
        else:
            ans_list.append(str(i))
    fin_ans = "".join(map(str, ans_list))
    fin_ans = fin_ans.replace("+-", "-")
    fin_ans = fin_ans.replace("*", "×")
    l = len(fin_ans)
    return fin_ans

def main(inp, max_num = 10**12):
    sp = lambda n: [str(n)[max(i-6,0):i] for i in range(len(str(n)), 0, -6)][::-1]
    num_s = sp(inp)

    n = sub_main(max_num**0.5)
    ans_lis = []
    l = len(num_s)
    for i, j in enumerate(num_s):
        ans = return_f(sub_main(abs(int(j))), int(j))
        ans_lis.append(ans)
        if i == l - 2:ans_lis.append(f"×{n}+")
        elif i != l - 1:
            num = l-i-1
            if num not in dic:
                make_dic(num)
                moji = num_last(num, dic)
            ans_lis.append(f"×({n})^{moji}+")

    return "".join(ans_lis)


def preprocess_expression(expr_str: str) -> str:
    processed_expr = expr_str.replace('×', '*')
    processed_expr = processed_expr.replace('^', '**')
    
    prev_expr = ""
    while prev_expr != processed_expr:
        prev_expr = processed_expr
        processed_expr = re.sub(r'\)\s*\(', ') * (', processed_expr)
        
    return processed_expr

def solve_arithmetic_expression(expr_str: str) -> float | int | str:
    processed_expr = preprocess_expression(expr_str)
    
    try:
        result = eval(processed_expr)
        return result
    except SyntaxError as se:
        return f"SyntaxError: {se} in processed expression: {processed_expr}"
    except Exception as e:
        return f"Error: {e} in processed expression: {processed_expr}"


def remove_unnecessary_parentheses(expr: str) -> str:
    """
    数式から不要な括弧を削除します。

    Args:
        expr: 括弧を含む数式文字列。

    Returns:
        不要な括弧が削除された数式文字列。
    """
    # Unicodeの乗算記号などを標準的な記号に置き換える
    expr = expr.replace('×', '*').replace('÷', '/')
    
    previous_expr = None
    # 式に変化がなくなるまでループ
    while expr != previous_expr:
        previous_expr = expr
        
        # 入れ子のない最も内側の括弧グループ `(...)` を探す
        match = re.search(r'\(([^()]+)\)', expr)
        
        if not match:
            break  # 括弧がなくなったら終了

        # 括弧全体 (例: "(-1-1+3+5)")
        full_match = match.group(0)
        # 括弧の中身 (例: "-1-1+3+5")
        content = match.group(1).strip()
        # 括弧の開始位置
        start_index = match.start()
        # 括弧の終了位置
        end_index = match.end()

        # --- 括弧を削除できるかどうかの判定 ---
        
        # ルール1: 中身が単純か？ (高優先度の演算子を含まないか)
        is_simple_content = '*' not in content and '/' not in content
        
        # ルール2: 括弧の直後に高優先度の演算子がないか？ (例: `(1+2)*3` のケース)
        is_followed_by_high_precedence = False
        if end_index < len(expr):
            following_char = expr[end_index:].strip()
            if following_char and following_char.startswith(('*', '/')):
                is_followed_by_high_precedence = True

        can_remove = False
        # 中身が単純で、計算順序を維持するための括弧でない場合
        if is_simple_content and not is_followed_by_high_precedence:
            can_remove = True
        
        # --- 処理の実行 ---

        if can_remove:
            # 括弧を削除して中身に置き換える
            expr = expr[:start_index] + " " + content + " " + expr[end_index:]
        else:
            # この括弧は削除できないため、一時的に別の文字に置き換えて
            # 次のループで再検索しないようにする
            expr = expr[:start_index] + '«' + content + '»' + expr[end_index:]

        # 連続する演算子や余分なスペースを整理
        expr = expr.replace('+ -', '- ').replace('--', '+ ')
        expr = expr.replace('( +', '(').replace('+  -', '- ')
        expr = re.sub(r'\s+', ' ', expr).strip()

    # 一時的に置き換えた文字を括弧に戻し、記号を元に戻す
    expr = expr.replace('«', '(').replace('»', ')')
    expr = expr.replace('*', '×').replace('/', '÷')
    
    return expr

def make_seive(t_num, max_n=10**12):
    global dic, num_1135_set, dic_1135, num_1135_str, num_1135, aa
    target_numbers = [int(i) for i in str(t_num)]
    dic_1135 = generate_and_evaluate_expressions(target_numbers)
    dic = {}
    num_1135 = [i for i, key in dic_1135.items()]
    num_1135_str = [str(i) for i in num_1135]  # 数値を文字列に変換
    num_1135_set = set(num_1135)  # 重複を排除
    num_1135 = num_1135[2:]  # 2以上の数値のみを使用
    num_1135.reverse()
    aa=fast_sieve(int(max_n**0.5)+1)

def all_main(n, max_n=10**12):
    global num
    num = n
    ll = main(num, max_n)

    c = solve_arithmetic_expression(ll)
    print(f"Evaluated result for {num}: {c} ok")
    return ll
use_number = 1135
make_seive(use_number)

def index_view(request):
    """
    メインページを表示するためのビュー
    """
    return render(request, 'calculator/index.html')

def calculate_view(request):
    """
    HTMXからのPOSTリクエストを処理し、計算結果を返すビュー
    """
    try:
        # POSTされてきた'number_input'の値を取得。空の場合は'0'とする
        number_str = request.POST.get('number_input', '0')
        # 念のため、空文字列が来た場合は0に変換
        if not number_str:
            number_str = '0'
        
        # 整数に変換して10倍する
        number = int(number_str)
        result = all_main(number)
    except (ValueError, TypeError):
        # 数字以外が入力された場合は0を返す
        result = 0
    
    # 計算結果をコンテキストに格納し、結果表示用の部分的なHTMLを返す
    context = {'result': result}
    return render(request, 'calculator/partials/result_partial.html', context)
