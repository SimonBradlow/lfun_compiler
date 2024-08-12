from typing import Set, Dict
import itertools
import sys
import traceback

from xxXXXX_support.python import *
import xxXXXX_support.x86 as x86
import constants
import cfun
from interference_graph import InterferenceGraph
import print_x86defs

comparisons = ['eq', 'gt', 'gte', 'lt', 'lte']
gensym_num = 0
global_logging = False

tuple_var_types = {}
function_names = set()

def log(label, value):
    if global_logging:
        print()
        print(f'--------------------------------------------------')
        print(f'Logging: {label}')
        print(value)
        print(f'--------------------------------------------------')


def log_ast(label, value):
    log(label, print_ast(value))


def gensym(x):
    """
    Constructs a new variable name guaranteed to be unique.
    :param x: A "base" variable name (e.g. "x")
    :return: A unique variable name (e.g. "x_1")
    """

    global gensym_num
    gensym_num = gensym_num + 1
    return f'{x}_{gensym_num}'


##################################################
# typecheck
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr]) | Begin(Stmts, Expr)
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

TEnv = Dict[str, type]

@dataclass
class Callable:
    args: List[type]
    output_type: type

def typecheck(program: Program) -> Program:
    """
    Typechecks the input program; throws an error if the program is not well-typed.
    :param program: The Lfun program to typecheck
    :return: The program, if it is well-typed
    """

    prim_arg_types = {
        'add':   [int, int],
        'sub':   [int, int],
        'mult':  [int, int],
        'not': [bool],
        'or':  [bool, bool],
        'and':  [bool, bool],
        'gt':   [int, int],
        'gte':  [int, int],
        'lt':   [int, int],
        'lte':  [int, int],
    }

    prim_output_types = {
        'add':   int,
        'sub':   int,
        'mult':  int,
        'not': bool,
        'or':  bool,
        'and':  bool,
        'gt':   bool,
        'gte':  bool,
        'lt':   bool,
        'lte':  bool,
    }

    def tc_exp(e: Expr, env: TEnv) -> type:
        match e:
            case Call(func, args):
                arg_types = [tc_exp(a, env) for a in args]
                match tc_exp(func, env):
                    case Callable(param_types, return_type):
                        assert param_types == arg_types
                        return return_type
                    case t:
                        raise Exception('expected function type, but got:', t)
            case Begin(stmts, e1):
                tc_stmts(stmts, env)
                return tc_exp(e1, env)
            case Prim('tuple', args):
                arg_types = [tc_exp(a, env) for a in args]
                t = tuple(arg_types)
                return t
            case Prim('subscript', [e1, Constant(i)]):
                t = tc_exp(e1, env)
                assert isinstance(t, tuple)
                return t[i]
            case Var(x):
                return env[x]
            case Constant(i):
                if isinstance(i, bool):
                    return bool
                elif isinstance(i, int):
                    return int
                else:
                    raise Exception('tc_exp', e)
            case Prim('eq', [e1, e2]):
                assert tc_exp(e1, env) == tc_exp(e2, env)
                return bool
            case Prim(op, args):
                arg_types = [tc_exp(a, env) for a in args]
                assert arg_types == prim_arg_types[op]
                return prim_output_types[op]
            case _:
                raise Exception('tc_exp', e)

    def tc_stmt(s: Stmt, env: TEnv):
        match s:
            case FunctionDef(name, params, body_stmts, return_type):
                function_names.add(name)
                arg_types = [t for x, t in params]
                env[name] = Callable(arg_types, return_type)
                new_env = env.copy()

                for x, t in params:
                    new_env[x] = t

                new_env['return value'] = return_type
                tc_stmts(body_stmts, new_env)

                for x in new_env:
                    if isinstance(new_env[x], tuple):
                        tuple_var_types[x] = new_env[x]

            case Return(e):
                assert env['return value'] == tc_exp(e, env)
            case While(condition, body_stmts):
                assert tc_exp(condition, env) == bool
                tc_stmts(body_stmts, env)
            case If(condition, then_stmts, else_stmts):
                assert tc_exp(condition, env) == bool
                tc_stmts(then_stmts, env)
                tc_stmts(else_stmts, env)
            case Print(e):
                tc_exp(e, env)
            case Assign(x, e):
                t_e = tc_exp(e, env)
                if x in env:
                    assert t_e == env[x]
                else:
                    env[x] = t_e
            case _:
                raise Exception('tc_stmt', s)

    def tc_stmts(stmts: List[Stmt], env: TEnv):
        for s in stmts:
            tc_stmt(s, env)

    env = {}
    tc_stmts(program.stmts, env)

    for x in env:
        if isinstance(env[x], tuple):
            tuple_var_types[x] = env[x]

    return program


##################################################
# remove-complex-opera*
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "tuple" | "subscript"
# Expr   ::= Var(x) | Constant(n) | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Expr, Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

def rco(prog: Program) -> Program:
    """
    Removes complex operands. After this pass, the arguments to operators (unary and binary
    operators, and function calls like "print") will be atomic.
    :param prog: An Lfun program
    :return: An Lfun program with atomic operator arguments.
    """

    def rco_stmt(stmt: Stmt, bindings: Dict[str, Expr]) -> Stmt:
        match stmt:
            case FunctionDef(name, params, body_stmts, return_type):
                return FunctionDef(name, params, rco_stmts(body_stmts), return_type)
            case Return(e):
                return Return(rco_exp(e, bindings))
            case Assign(x, e1):
                new_e1 = rco_exp(e1, bindings)
                return Assign(x, new_e1)
            case Print(e1):
                new_e1 = rco_exp(e1, bindings)
                return Print(new_e1)
            case While(condition, body_stmts):
                condition_bindings = {}
                condition_exp = rco_exp(condition, condition_bindings)
                condition_stmts = [Assign(x, e) for x, e in condition_bindings.items()]
                new_condition = Begin(condition_stmts, condition_exp)

                new_body_stmts = rco_stmts(body_stmts)
                return While(new_condition, new_body_stmts)

            case If(condition, then_stmts, else_stmts):
                new_condition = rco_exp(condition, bindings)
                new_then_stmts = rco_stmts(then_stmts)
                new_else_stmts = rco_stmts(else_stmts)

                return If(new_condition,
                          new_then_stmts,
                          new_else_stmts)
            case _:
                raise Exception('rco_stmt', stmt)

    def rco_stmts(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []

        for stmt in stmts:
            bindings = {}
            # (1) compile the statement
            new_stmt = rco_stmt(stmt, bindings)
            # (2) add the new bindings created by rco_exp
            new_stmts.extend([Assign(x, e) for x, e in bindings.items()])
            # (3) add the compiled statement itself
            new_stmts.append(new_stmt)

        return new_stmts

    def rco_exp(e: Expr, bindings: Dict[str, Expr]) -> Expr:
        match e:
            case Call(func, args):
                new_args = [rco_exp(e, bindings) for e in args]
                new_func = rco_exp(func, bindings)
                new_e = Call(new_func, new_args)
                new_v = gensym('tmp')
                bindings[new_v] = new_e
                return Var(new_v)
            case Var(x):
                if x in function_names:
                    new_v = gensym('tmp')
                    bindings[new_v] = Var(x)
                    return Var(new_v)
                else:
                    return Var(x)
            case Constant(i):
                return Constant(i)
            case Prim(op, args):
                new_args = [rco_exp(e, bindings) for e in args]
                new_e = Prim(op, new_args)
                new_v = gensym('tmp')
                bindings[new_v] = new_e
                return Var(new_v)
            case _:
                raise Exception('rco_exp', e)

    return Program(rco_stmts(prog.stmts))


##################################################
# convert_closures
##################################################

def convert_closures(prog: Program) -> Program:

    def free_vars_exp(e: Expr, bound_vars: List[Expr], func_args: List[Expr]) -> List[Expr]:
        match e:
            case Call(func, args): 
                var_list = []
                var_list = [free_vars_exp(e, bound_vars, func_args) for e in args]
                return var_list
            case Var(x):
                if (Var(x) not in bound_vars) and (Var(x) not in func_args):
                    return [Var(x)]
                else:
                    return []
            case Constant(i):
                return []
            case Prim(op, args):
                var_list = []
                var_list = [free_vars_exp(e, bound_vars, func_args) for e in args]
                return var_list
            case _:
                raise Exception('fv_exp', e)

    def free_vars_stmt(stmt: Stmt, bound_vars: List[Expr], func_args: List[Expr]) -> List[Expr]:
        match stmt:
            case FunctionDef(name, params, body_stmts, return_type):
                return []
            case Return(e):
                return free_vars_exp(e, bound_vars, func_args)
            case Assign(x, e1):
                return free_vars_exp(e1, bound_vars, func_args)
            case Print(e1):
                return free_vars_exp(e1, bound_vars, func_args)
            case While(condition, body_stmts):
                return []
            case If(condition, then_stmts, else_stmts):
                return []
            case _:
                raise Exception('fv_stmt', stmt)

    def free_vars_stmts(stmts: List[Stmt], params: List[Expr]) -> List[Expr]:
        free_vars = []
        for stmt in stmts:
            new_vars = free_vars_stmt(stmt, free_vars, params)
            if new_vars not in free_vars:
                free_vars.extend(new_vars)
        return free_vars

    def lift_nested_functions(stmts: List[Stmt]) -> List[Stmt]:
        new_stmts = []
        for stmt in reversed(stmts):
            match stmt:
                case FunctionDef(name, params, body_stmts, return_type):
                    new_body = []
                    inner_function = None
                    for s in body_stmts:
                        match s:
                            case FunctionDef(name2, params2, body_stmts2, return_type2):
                                # When you find a funciton def, rename it so you can use the original name for the closure, and lift it to the top-level
                                new_name = name2+'_fun'

                                # Add a new argument to the lifted funciton: the "closure object" (a tuple), which contains all the free variables of the function
                                new_params = params2.copy()
                                new_params.append(('clos', (name, int)))

                                # closure object creation
                                closure_object = free_vars_stmts(body_stmts2, params2)

                                # Modify the body of the function to restore the values of the free variables from the closure object
                                new_body2 = []
                                for obj in closure_object:
                                    new_body2.append(Assign(obj.name, Prim('subscript', [Var('clos'), Constant(0)])))
                                new_body2.extend(body_stmts2)
                                
                                # fix to typecheck the newly added closure tuple
                                type_list = []
                                for obj in closure_object:
                                    type_list.append(type(obj))
                                tuple_var_types[name2] = tuple(type_list)
                                
                                # Function to be lifted
                                inner_function = FunctionDef(new_name, new_params, new_body2, return_type2)
                                
                                # Where the funciton was defined, instead leave a closure object
                                closure_object.insert(0, Var(new_name))
                                new_body.append(Assign(name2, Prim('tuple', closure_object)))

                            case Call(func, args):
                                # Replace `Call(Var(f), args)` with `Call(Prim('subscript', [Var(f), Constant(0)]), args + [Var(f)])`
                                new_body.append(Call(Prim('subscript', [func, Constant(0)]), args + [func]))
                            case _:
                                new_body.append(s)
                    new_stmts.append(FunctionDef(name, params, new_body, return_type))
                    if inner_function is not None:
                        new_stmts.append(inner_function)
                case _:
                    new_stmts.append(stmt)
                
        return list(reversed(new_stmts))
    

    # Hack fix to avoid true recursion. I was getting confused.
    # And if anything, this has less permutations that a recursive function.
    input_stmts = prog.stmts
    return_prog = lift_nested_functions(input_stmts)
    while (return_prog != input_stmts):
        input_stmts = return_prog
        return_prog = lift_nested_functions(input_stmts)

    return Program(return_prog)


##################################################
# explicate-control
##################################################
# op     ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#          | "subscript" | "allocate" | "collect" | "tuple_set"
# Atm    ::= Var(x) | Constant(n)
# Expr   ::= Atm | Prim(op, List[Expr])
#          | Call(Expr, List[Expr])
# Stmt   ::= Assign(x, Expr) | Print(Expr) | If(Expr, Stmts, Stmts) | While(Begin(Stmts, Expr), Stmts)
#          | Return(Expr) | FunctionDef(str, List[Tuple[str, type]], List[Stmt], type)
# Stmts  ::= List[Stmt]
# LFun   ::= Program(Stmts)

def explicate_control(prog: Program) -> cfun.CProgram:
    """
    Transforms an Lfun Expression into a Cif program.
    :param prog: An Lfun Expression
    :return: A Cif Program
    """

    regular_stmts = []
    function_defs = []

    for s in prog.stmts:
        match s:
            case FunctionDef(name, params, body_stmts, return_type):
                blocks = _explicate_control(name, Program(body_stmts))
                param_names = [a[0] for a in params]
                function_defs.append(cfun.CFunctionDef(name, param_names, blocks))
            case _:
                regular_stmts.append(s)

    main_blocks = _explicate_control('main', Program(regular_stmts))
    function_defs.append(cfun.CFunctionDef('main', [], main_blocks))

    return cfun.CProgram(function_defs)

def _explicate_control(current_function: str, prog: Program) -> cfun.CProgram:
    """
    Transforms an Lfun Expression into a Cif program.
    :param prog: An Lfun Expression
    :return: A Cif Program
    """

    # the basic blocks of the program
    basic_blocks: Dict[str, List[cfun.Stmt]] = {}

    # create a new basic block to hold some statements
    # generates a brand-new name for the block and returns it
    def create_block(stmts: List[cfun.Stmt]) -> str:
        label = gensym('label')
        basic_blocks[label] = stmts
        return label

    def explicate_atm(e: Expr) -> cfun.Atm:
        match e:
            case Var(x):
                return cfun.Var(x)
            case Constant(c):
                return cfun.Constant(c)
            case _:
                raise RuntimeError(e)

    def explicate_exp(e: Expr) -> cfun.Expr:
        match e:
            case Call(func, args):
                new_args = [explicate_atm(a) for a in args]
                return cfun.Call(explicate_atm(func), new_args)
            case Prim(op, args):
                new_args = [explicate_atm(a) for a in args]
                return cfun.Prim(op, new_args)
            case _:
                return explicate_atm(e)

    def explicate_stmt(stmt: Stmt, cont: List[cfun.Stmt]) -> List[cfun.Stmt]:
        match stmt:
            case Return(e):
                # discard the continuation!
                new_exp = explicate_atm(e)
                return [cfun.Return(new_exp)]
            case Assign(x, exp):
                new_exp = explicate_exp(exp)
                new_stmt: List[cfun.Stmt] = [cfun.Assign(x, new_exp)]
                return new_stmt + cont
            case Print(exp):
                new_exp = explicate_atm(exp)
                new_stmt: List[cfun.Stmt] = [cfun.Print(new_exp)]
                return new_stmt + cont
            case While(Begin(condition_stmts, condition_exp), body_stmts):
                cont_label = create_block(cont)
                test_label = gensym('loop_label')
                body_label = create_block(explicate_stmts(body_stmts, [cfun.Goto(test_label)]))
                test_stmts = [cfun.If(explicate_exp(condition_exp),
                                     cfun.Goto(body_label),
                                     cfun.Goto(cont_label))]
                basic_blocks[test_label] = explicate_stmts(condition_stmts, test_stmts)
                return [cfun.Goto(test_label)]
            case If(condition, then_stmts, else_stmts):
                cont_label = create_block(cont)
                e2_label = create_block(explicate_stmts(then_stmts, [cfun.Goto(cont_label)]))
                e3_label = create_block(explicate_stmts(else_stmts, [cfun.Goto(cont_label)]))
                return [cfun.If(explicate_exp(condition),
                               cfun.Goto(e2_label),
                               cfun.Goto(e3_label))]
                
            case _:
                raise RuntimeError(stmt)

    def explicate_stmts(stmts: List[Stmt], cont: List[cfun.Stmt]) -> List[cfun.Stmt]:
        for s in reversed(stmts):
            cont = explicate_stmt(s, cont)
        return cont

    new_body = [cfun.Return(cfun.Constant(0))]
    new_body = explicate_stmts(prog.stmts, new_body)

    basic_blocks[current_function + 'start'] = new_body
    return basic_blocks


##################################################
# select-instructions
##################################################
# op           ::= "add" | "sub" | "mult" | "not" | "or" | "and" | "eq" | "gt" | "gte" | "lt" | "lte"
#                | "subscript" | "allocate" | "collect" | "tuple_set"
# Atm          ::= Var(x) | Constant(n)
# Expr         ::= Atm | Prim(op, List[Expr])
# Stmt         ::= Assign(x, Expr) | Print(Expr)
#                | If(Expr, Goto(label), Goto(label)) | Goto(label) | Return(Expr)
# Stmts        ::= List[Stmt]
# CFunctionDef ::= CFunctionDef(name, List[str], Dict[label, Stmts])
# Cif         ::= CProgram(List[CFunctionDef])

@dataclass(frozen=True, eq=True)
class X86FunctionDef(AST):
    label: str
    blocks: Dict[str, List[x86.Instr]]
    stack_space: Tuple[int, int]

@dataclass(frozen=True, eq=True)
class X86ProgramDefs(AST):
    defs: List[X86FunctionDef]


def select_instructions(prog: cfun.CProgram) -> X86ProgramDefs:
    """
    Transforms a Lfun program into a pseudo-x86 assembly program.
    :param prog: a Lfun program
    :return: a pseudo-x86 program
    """

    function_defs = []

    for d in prog.defs:
        match d:
            case cfun.CFunctionDef(name, args, blocks):
                p = _select_instructions(name, d)
                setup_instrs = [x86.NamedInstr('movq', [x86.Reg(r), x86.Var(a)]) \
                                for a, r in zip(args, constants.argument_registers)]
                p.blocks[name + 'start'] = setup_instrs + p.blocks[name + 'start']

                function_defs.append(X86FunctionDef(name, p.blocks, None))
    return X86ProgramDefs(function_defs)


def _select_instructions(current_function: str, prog: cfun.CProgram) -> x86.X86Program:
    """
    Transforms a Lfun program into a pseudo-x86 assembly program.
    :param prog: a Lfun program
    :return: a pseudo-x86 program
    """

    def mk_tag(types: Tuple[type]) -> int:
        """
        Builds a vector tag. See section 5.2.2 in the textbook.
        :param types: A list of the types of the vector's elements.
        :return: A vector tag, as an integer.
        """
        pointer_mask = 0
        # for each type in the vector, encode it in the pointer mask
        for t in reversed(types):
            # shift the mask by 1 bit to make room for this type
            pointer_mask = pointer_mask << 1

            if isinstance(t, tuple):
                # if it's a vector type, the mask is 1
                pointer_mask = pointer_mask + 1
            else:
                # otherwise, the mask is 0 (do nothing)
                pass

        # shift the pointer mask by 6 bits to make room for the length field
        mask_and_len = pointer_mask << 6
        mask_and_len = mask_and_len + len(types) # add the length

        # shift the mask and length by 1 bit to make room for the forwarding bit
        tag = mask_and_len << 1
        tag = tag + 1

        return tag

    def si_atm(a: cfun.Expr) -> x86.Arg:
        match a:
            case cfun.Constant(i):
                return x86.Immediate(int(i))
            case cfun.Var(x):
                return x86.Var(x)
            case _:
                raise Exception('si_atm', a)

    def si_stmts(stmts: List[cfun.Stmt]) -> List[x86.Instr]:
        instrs = []

        for stmt in stmts:
            instrs.extend(si_stmt(stmt))

        return instrs

    op_cc = {'eq': 'e', 'gt': 'g', 'gte': 'ge', 'lt': 'l', 'lte': 'le'}

    binop_instrs = {'add': 'addq', 'sub': 'subq', 'mult': 'imulq', 'and': 'andq', 'or': 'orq'}

    def si_stmt(stmt: cfun.Stmt) -> List[x86.Instr]:
        match stmt:
            case cfun.Assign(x, cfun.Var(f)) if f in function_names:
                return [x86.NamedInstr('leaq', [x86.GlobalVal(f), x86.Var(x)])]
            case cfun.Assign(x, cfun.Call(fun, args)):
                instrs = []

                # save caller-saved registers
                for r in constants.caller_saved_registers:
                    instrs += [x86.NamedInstr('pushq', [x86.Reg(r)])]

                # place arguments in argument registers
                for a, r in zip(args, constants.argument_registers):
                    instrs += [x86.NamedInstr('movq', [si_atm(a), x86.Reg(r)])]

                # call the function
                instrs += [x86.IndirectCallq(si_atm(fun), 0)]

                # restore caller-saved registers
                for r in reversed(constants.caller_saved_registers):
                    instrs += [x86.NamedInstr('popq', [x86.Reg(r)])]

                # move the result from rax into the destination
                instrs += [x86.NamedInstr('movq', [x86.Reg('rax'), x86.Var(x)])]
                return instrs
            case cfun.Assign(x, cfun.Prim('tuple', args)):
                tag = mk_tag(tuple_var_types[x])
                instrs = [x86.NamedInstr('movq', [x86.Immediate(8*(1+len(args))), x86.Reg('rdi')]),
                          x86.Callq('allocate'),
                          x86.NamedInstr('movq', [x86.Reg('rax'), x86.Reg('r11')]),
                          x86.NamedInstr('movq', [x86.Immediate(tag), x86.Deref('r11', 0)])]
                for i, a in enumerate(args):
                    instrs.append(x86.NamedInstr('movq', [si_atm(a), x86.Deref('r11', 8*(i+1))]))
                instrs.append(x86.NamedInstr('movq', [x86.Reg('r11'), x86.Var(x)]))
                return instrs
            case cfun.Assign(x, cfun.Prim('subscript', [atm1, cfun.Constant(idx)])):
                offset_bytes = 8 * (idx + 1)
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('r11')]),
                        x86.NamedInstr('movq', [x86.Deref('r11', offset_bytes), x86.Var(x)])]
            case cfun.Assign(x, cfun.Prim(op, [atm1, atm2])):
                if op in binop_instrs:
                    return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                            x86.NamedInstr(binop_instrs[op], [si_atm(atm2), x86.Reg('rax')]),
                            x86.NamedInstr('movq', [x86.Reg('rax'), x86.Var(x)])]
                elif op in op_cc:
                    return [x86.NamedInstr('cmpq', [si_atm(atm2), si_atm(atm1)]),
                            x86.Set(op_cc[op], x86.ByteReg('al')),
                            x86.NamedInstr('movzbq', [x86.ByteReg('al'), x86.Var(x)])]

                else:
                    raise Exception('si_stmt failed op', op)
            case cfun.Assign(x, cfun.Prim('not', [atm1])):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Var(x)]),
                        x86.NamedInstr('xorq', [x86.Immediate(1), x86.Var(x)])]
            case cfun.Assign(x, atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Var(x)])]
            case cfun.Print(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rdi')]),
                        x86.Callq('print_int')]
            case cfun.Return(atm1):
                return [x86.NamedInstr('movq', [si_atm(atm1), x86.Reg('rax')]),
                        x86.Jmp(current_function + 'conclusion')]
            case cfun.Goto(label):
                return [x86.Jmp(label)]
            case cfun.If(a, cfun.Goto(then_label), cfun.Goto(else_label)):
                return [x86.NamedInstr('cmpq', [si_atm(a), x86.Immediate(1)]),
                        x86.JmpIf('e', then_label),
                        x86.Jmp(else_label)]
            case _:
                raise Exception('si_stmt', stmt)

    basic_blocks = {label: si_stmts(block) for (label, block) in prog.blocks.items()}
    return x86.X86Program(basic_blocks)


##################################################
# allocate-registers
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

Color = int
Coloring = Dict[x86.Var, Color]
Saturation = Set[Color]

def allocate_registers(program: X86ProgramDefs) -> X86ProgramDefs:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param program: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """

    new_defs = []
    for d in program.defs:
        new_program = _allocate_registers(d.label, x86.X86Program(d.blocks))
        new_defs.append(X86FunctionDef(d.label, new_program.blocks, new_program.stack_space))
    return X86ProgramDefs(new_defs)

def _allocate_registers(current_function: str, program: x86.X86Program) -> x86.X86Program:
    """
    Assigns homes to variables in the input program. Allocates registers and
    stack locations as needed, based on a graph-coloring register allocation
    algorithm.
    :param program: A pseudo-x86 program.
    :return: An x86 program, annotated with the number of bytes needed in stack
    locations.
    """

    blocks = program.blocks
    live_before_sets = {current_function + 'conclusion': set()}
    for label in blocks:
        live_before_sets[label] = set()

    live_after_sets = {}
    homes: Dict[x86.Var, x86.Arg] = {}
    tuple_homes: Dict[x86.Var, x86.Arg] = {}

    tuple_vars = set(tuple_var_types.keys())

    # --------------------------------------------------
    # utilities
    # --------------------------------------------------
    def vars_arg(a: x86.Arg) -> Set[x86.Var]:
        match a:
            case x86.GlobalVal(x):
                return set()
            case x86.Immediate(i):
                return set()
            case x86.Reg(r):
                return set()
            case x86.ByteReg(r):
                return set()
            case x86.Var(x):
                if x in tuple_var_types:
                    return set()
                else:
                    return {x86.Var(x)}
            case x86.Deref(r, offset):
                return set()
            case _:
                raise Exception('ul_arg', a)

    def reads_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, [e1]) if i in ['pushq', 'popq']:
                assert isinstance(e1, x86.Reg)
                return set()
            case x86.IndirectCallq(e1):
                return vars_arg(e1)
            case x86.NamedInstr(i, [e1, e2]) if i in ['movq', 'movzbq']:
                return vars_arg(e1)
            case x86.NamedInstr(i, [e1, e2]) if i in ['addq', 'subq', 'imulq', 'cmpq', 'andq', 'orq', 'xorq', 'leaq']:
                return vars_arg(e1).union(vars_arg(e2))
            case x86.Jmp(label) | x86.JmpIf(_, label):
                # the variables that might be read after this instruction
                # are the live-before variables of the destination block
                return live_before_sets[label]
            case _:
                if isinstance(i, (x86.Callq, x86.Set)):
                    return set()
                else:
                    raise Exception(i)

    def writes_of(i: x86.Instr) -> Set[x86.Var]:
        match i:
            case x86.NamedInstr(i, [e1]) if i in ['pushq', 'popq']:
                assert isinstance(e1, x86.Reg)
                return set()
            case x86.NamedInstr(i, [e1, e2]) \
                if i in ['movq', 'movzbq', 'addq', 'subq', 'imulq', 'cmpq', 'andq', 'orq', 'xorq', 'leaq']:
                return vars_arg(e2)
            case _:
                if isinstance(i, (x86.Jmp, x86.JmpIf, x86.Callq, x86.Set, x86.IndirectCallq)):
                    return set()
                else:
                    raise Exception(i)

    # --------------------------------------------------
    # liveness analysis
    # --------------------------------------------------
    def ul_instr(i: x86.Instr, live_after: Set[x86.Var]) -> Set[x86.Var]:
        return live_after.difference(writes_of(i)).union(reads_of(i))

    def ul_block(label: str):
        instrs = blocks[label]
        current_live_after: Set[x86.Var] = set()

        block_live_after_sets = []
        for i in reversed(instrs):
            block_live_after_sets.append(current_live_after)
            current_live_after = ul_instr(i, current_live_after)

        live_before_sets[label] = current_live_after
        live_after_sets[label] = list(reversed(block_live_after_sets))

    def ul_fixpoint(labels: List[str]):
        fixpoint_reached = False

        while not fixpoint_reached:
            old_live_befores = live_before_sets.copy()

            for label in labels:
                ul_block(label)

            if old_live_befores == live_before_sets:
                fixpoint_reached = True

    # --------------------------------------------------
    # interference graph
    # --------------------------------------------------
    def bi_instr(e: x86.Instr, live_after: Set[x86.Var], graph: InterferenceGraph):
        for v1 in writes_of(e):
            for v2 in live_after:
                graph.add_edge(v1, v2)

    def bi_block(instrs: List[x86.Instr], live_afters: List[Set[x86.Var]], graph: InterferenceGraph):
        for instr, live_after in zip(instrs, live_afters):
            bi_instr(instr, live_after, graph)

    # --------------------------------------------------
    # graph coloring
    # --------------------------------------------------
    def color_graph(local_vars: Set[x86.Var], interference_graph: InterferenceGraph) -> Coloring:
        coloring: Coloring = {}

        to_color = local_vars.copy()

        # Saturation sets start out empty
        saturation_sets = {x: set() for x in local_vars}

        # Loop until we are finished coloring
        while to_color:
            # Find the variable x with the largest saturation set
            x = max(to_color, key=lambda x: len(saturation_sets[x]))

            # Remove x from the variables to color
            to_color.remove(x)

            # Find the smallest color not in x's saturation set
            x_color = next(i for i in itertools.count() if i not in saturation_sets[x])

            # Assign x's color
            coloring[x] = x_color

            # Add x's color to the saturation sets of its neighbors
            for y in interference_graph.neighbors(x):
                saturation_sets[y].add(x_color)

        return coloring

    # --------------------------------------------------
    # assigning homes
    # --------------------------------------------------
    def align(num_bytes: int) -> int:
        if num_bytes % 16 == 0:
            return num_bytes
        else:
            return num_bytes + (16 - (num_bytes % 16))

    def ah_arg(a: x86.Arg) -> x86.Arg:
        match a:
            case x86.GlobalVal(x):
                return a
            case x86.Immediate(i):
                return a
            case x86.Deref(r, offset):
                return a
            case x86.Reg(r):
                return a
            case x86.ByteReg(r):
                return a
            case x86.Var(x):
                if x in tuple_vars:
                    if x in tuple_homes:
                        return tuple_homes[x]
                    else:
                        current_stack_size = len(tuple_homes) * 8
                        offset = -(current_stack_size + 8)
                        tuple_homes[x] = x86.Deref('r15', offset)
                        return x86.Deref('r15', offset)
                else:
                    if a in homes:
                        return homes[a]
                    else:
                        return x86.Reg('r8')
            case _:
                raise Exception('ah_arg', a)

    def ah_instr(e: x86.Instr) -> x86.Instr:
        match e:
            case x86.IndirectCallq(a1, n):
                return x86.IndirectCallq(ah_arg(a1), n)
            case x86.NamedInstr(i, args):
                return x86.NamedInstr(i, [ah_arg(a) for a in args])
            case x86.Set(cc, a1):
                return x86.Set(cc, ah_arg(a1))
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf)):
                    return e
                else:
                    raise Exception('ah_instr', e)

    def ah_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        return [ah_instr(i) for i in instrs]

    # --------------------------------------------------
    # main body of the pass
    # --------------------------------------------------

    # Step 1: Perform liveness analysis
    all_labels = list(blocks.keys())
    ul_fixpoint(all_labels)
    log_ast('live-after sets', live_after_sets)

    # Step 2: Build the interference graph
    interference_graph = InterferenceGraph()

    for label in blocks.keys():
        bi_block(blocks[label], live_after_sets[label], interference_graph)

    log_ast('interference graph', interference_graph)

    # Step 3: Color the graph
    all_vars = interference_graph.get_nodes()
    coloring = color_graph(all_vars, interference_graph)
    colors_used = set(coloring.values())
    log('coloring', coloring)

    # Defines the set of registers to use
    available_registers = constants.caller_saved_registers + constants.callee_saved_registers

    # Step 4: map variables to homes
    color_map = {}
    stack_locations_used = 0

    # Step 4.1: Map colors to locations (the "color map")
    for color in sorted(colors_used):
        if available_registers != []:
            r = available_registers.pop()
            color_map[color] = x86.Reg(r)
        else:
            offset = stack_locations_used+1
            color_map[color] = x86.Deref('rbp', -(offset * 8))
            stack_locations_used += 1

    # Step 4.2: Compose the "coloring" with the "color map" to get "homes"
    for v in all_vars:
        homes[v] = color_map[coloring[v]]
    log('homes', homes)

    # Step 5: replace variables with their homes
    blocks = program.blocks
    new_blocks = {label: ah_block(block) for label, block in blocks.items()}
    regular_stack_space = align(8 * stack_locations_used)
    root_stack_slots = len(tuple_homes)
    return x86.X86Program(new_blocks, stack_space = (regular_stack_space, root_stack_slots))


##################################################
# patch-instructions
##################################################
# Arg            ::= Immediate(i) | Reg(r) | ByteReg(r) | Var(x) | Deref(r, offset) | GlobalVal(x)
# op             ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#                  | 'leaq'
# cc             ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr          ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#                  | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)
#                  | IndirectCallq(Arg)
# Blocks         ::= Dict[label, List[Instr]]
# X86FunctionDef ::= X86FunctionDef(name, Blocks)
# X86ProgramDefs ::= List[X86FunctionDef]

def patch_instructions(program: X86ProgramDefs) -> X86ProgramDefs:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param program: An x86 program.
    :return: A patched x86 program.
    """

    new_defs = []
    for d in program.defs:
        new_prog = _patch_instructions(x86.X86Program(d.blocks))
        new_defs.append(X86FunctionDef(d.label, new_prog.blocks, d.stack_space))
    return X86ProgramDefs(new_defs)

def _patch_instructions(program: x86.X86Program) -> x86.X86Program:
    """
    Patches instructions with two memory location inputs, using %rax as a temporary location.
    :param program: An x86 program.
    :return: A patched x86 program.
    """

    def pi_instr(e: x86.Instr) -> List[x86.Instr]:
        match e:
            case x86.NamedInstr(i, [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                return [x86.NamedInstr('movq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                        x86.NamedInstr(i, [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Deref(r2, o2)]):
                    return [x86.NamedInstr('movzbq', [x86.Deref(r1, o1), x86.Reg('rax')]),
                            x86.NamedInstr('movq', [x86.Reg('rax'), x86.Deref(r2, o2)])]
            case x86.NamedInstr('cmpq', [a1, x86.Immediate(i)]):
                return [x86.NamedInstr('movq', [x86.Immediate(i), x86.Reg('rax')]),
                        x86.NamedInstr('cmpq', [a1, x86.Reg('rax')])]
            case _:
                if isinstance(e, (x86.Callq, x86.Retq, x86.Jmp, x86.JmpIf, x86.NamedInstr,
                                  x86.Set, x86.IndirectCallq)):
                    return [e]
                else:
                    raise Exception('pi_instr', e)

    def pi_block(instrs: List[x86.Instr]) -> List[x86.Instr]:
        new_instrs = [pi_instr(i) for i in instrs]
        flattened = [val for sublist in new_instrs for val in sublist]
        return flattened

    blocks = program.blocks
    new_blocks = {label: pi_block(block) for label, block in blocks.items()}
    return x86.X86Program(new_blocks, stack_space = program.stack_space)


##################################################
# prelude-and-conclusion
##################################################
# Arg    ::= Immediate(i) | Reg(r) | ByteReg(r) | Deref(r, offset) | GlobalVal(x)
# op     ::= 'addq' | 'subq' | 'imulq' | 'cmpq' | 'andq' | 'orq' | 'xorq' | 'movq' | 'movzbq'
#          | 'leaq'
# cc     ::= 'e'| 'g' | 'ge' | 'l' | 'le'
# Instr  ::= NamedInstr(op, List[Arg]) | Callq(label) | Retq()
#          | Jmp(label) | JmpIf(cc, label) | Set(cc, Arg)

#          | IndirectCallq(Arg)
# Blocks ::= Dict[label, List[Instr]]
# X86    ::= X86Program(Blocks)

def prelude_and_conclusion(program: X86ProgramDefs) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param program: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """
    all_blocks = {}
    for d in program.defs:
        new_prog = _prelude_and_conclusion(d.label, x86.X86Program(d.blocks, d.stack_space))
        for label, instrs in new_prog.blocks.items():
            all_blocks[label] = instrs
    return x86.X86Program(all_blocks)

def _prelude_and_conclusion(current_function: str, program: x86.X86Program) -> x86.X86Program:
    """
    Adds the prelude and conclusion for the program.
    :param program: An x86 program.
    :return: An x86 program, with prelude and conclusion.
    """
    stack_bytes, root_stack_locations = program.stack_space

    # Prelude
    prelude = [x86.NamedInstr('pushq', [x86.Reg('rbp')]),
               x86.NamedInstr('movq',  [x86.Reg('rsp'), x86.Reg('rbp')])]

    for r in constants.callee_saved_registers:
        prelude += [x86.NamedInstr('pushq', [x86.Reg(r)])]

    prelude += [x86.NamedInstr('subq',  [x86.Immediate(stack_bytes),
                                        x86.Reg('rsp')])]

    if current_function == 'main':
        prelude += [x86.NamedInstr('movq',  [x86.Immediate(constants.root_stack_size),
                                             x86.Reg('rdi')]),
                    x86.NamedInstr('movq',  [x86.Immediate(constants.heap_size),
                                            x86.Reg('rsi')]),
                    x86.Callq('initialize'),
                    x86.NamedInstr('movq',  [x86.GlobalVal('rootstack_begin'), x86.Reg('r15')])]

    for _ in range(root_stack_locations):
        prelude += [x86.NamedInstr('movq', [x86.Immediate(0), x86.Deref('r15', 0)]),
                    x86.NamedInstr('addq', [x86.Immediate(8), x86.Reg('r15')])]
    prelude += [x86.Jmp(current_function + 'start')]

    # Conclusion
    conclusion = [x86.NamedInstr('addq', [x86.Immediate(stack_bytes), x86.Reg('rsp')]),
                  x86.NamedInstr('subq', [x86.Immediate(8*root_stack_locations), x86.Reg('r15')])]

    for r in reversed(constants.callee_saved_registers):
        conclusion += [x86.NamedInstr('popq', [x86.Reg(r)])]

    conclusion += [x86.NamedInstr('popq', [x86.Reg('rbp')]),
                   x86.Retq()]

    new_blocks = program.blocks.copy()
    new_blocks[current_function] = prelude
    new_blocks[current_function + 'conclusion'] = conclusion
    return x86.X86Program(new_blocks, stack_space = program.stack_space)


##################################################
# add-allocate
##################################################

def add_allocate(program: str) -> str:
    """
    Adds the 'allocate' function to the end of the program.
    :param program: An x86 program, as a string.
    :return: An x86 program, as a string, with the 'allocate' function.
    """

    alloc = """
allocate:
  movq free_ptr(%rip), %rax
  addq %rdi, %rax
  movq %rdi, %rsi
  cmpq fromspace_end(%rip), %rax
  jl allocate_alloc
  movq %r15, %rdi
  callq collect
  jmp allocate_alloc
allocate_alloc:
  movq free_ptr(%rip), %rax
  addq %rsi, free_ptr(%rip)
  retq
"""
    return program + alloc

##################################################
# Compiler definition
##################################################

compiler_passes = {
    'typecheck': typecheck,
    'remove complex opera*': rco,
    'typecheck2': typecheck,
    'convert closures': convert_closures,
    'explicate control': explicate_control,
    'select instructions': select_instructions,
    'allocate registers': allocate_registers,
    'patch instructions': patch_instructions,
    'prelude & conclusion': prelude_and_conclusion,
    'print x86': x86.print_x86,
    'add allocate': add_allocate
}


def run_compiler(s, logging=False):
    global tuple_var_types, function_names
    tuple_var_types = {}
    function_names = set()

    def print_prog(current_program):
        print('Concrete syntax:')
        if isinstance(current_program, x86.X86Program):
            print(x86.print_x86(current_program))
        elif isinstance(current_program, X86ProgramDefs):
            print(print_x86defs.print_x86_defs(current_program))
        elif isinstance(current_program, Program):
            print(print_program(current_program))
        elif isinstance(current_program, cfun.CProgram):
            print(cfun.print_program(current_program))

        print()
        print('Abstract syntax:')
        print(print_ast(current_program))

    current_program = parse(s)

    if logging == True:
        print()
        print('==================================================')
        print(' Input program')
        print('==================================================')
        print()
        print_prog(current_program)

    for pass_name, pass_fn in compiler_passes.items():
        current_program = pass_fn(current_program)

        if logging == True:
            print()
            print('==================================================')
            print(f' Output of pass: {pass_name}')
            print('==================================================')
            print()
            print_prog(current_program)

    return current_program


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python compiler.py <source filename>')
    else:
        file_name = sys.argv[1]
        with open(file_name) as f:
            print(f'Compiling program {file_name}...')

            try:
                program = f.read()
                x86_program = run_compiler(program, logging=True)

                with open(file_name + '.s', 'w') as output_file:
                    output_file.write(x86_program)

            except:
                print('Error during compilation! **************************************************')
                traceback.print_exception(*sys.exc_info())


