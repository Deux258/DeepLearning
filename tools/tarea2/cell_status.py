"""Añade confirmaciones visibles al código; no es una dependencia de los notebooks."""
import ast


MARKER = '# TAREA2_CELL_STATUS'


def without_completion_message(source):
    """Retira únicamente nuestra confirmación final para poder volver a numerarla."""
    before, marker, tail = source.rpartition('\n' + MARKER + '\n')
    if not marker:
        return source
    nodes = ast.parse(tail).body
    if not (len(nodes) == 1 and isinstance(nodes[0], ast.Expr)
            and isinstance(nodes[0].value, ast.Call)
            and isinstance(nodes[0].value.func, ast.Name)
            and nodes[0].value.func.id == 'print'
            and nodes[0].value.args
            and isinstance(nodes[0].value.args[0], ast.Constant)
            and isinstance(nodes[0].value.args[0].value, str)
            and nodes[0].value.args[0].value.startswith('[OK] Celda de código ')):
        raise ValueError('Hay código adicional después del marcador de confirmación; revisarlo antes de regenerar.')
    return before.rstrip() + '\n'


def add_completion_messages(notebook):
    """Un print al terminar cada celda; una excepción impide alcanzar ese print."""
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    for index, cell in enumerate(code_cells, 1):
        source = without_completion_message(''.join(cell['source'])).rstrip()
        message = f'[OK] Celda de código {index:02d}/{len(code_cells):02d} finalizada sin errores.'
        source += f'\n\n{MARKER}\nprint({message!r}, flush=True)\n'
        ast.parse(source)
        cell['source'] = source.splitlines(keepends=True)
