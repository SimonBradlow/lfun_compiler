### Anonymous (Lexically-scoped) Functions

This project was my final project for a class in Compiler Construction, and thus, the necessary support code for compilation has been redacted at the request of my professor. Other support code and class files can be found within the repo.

*ALL CODE WRITTEN BY ME, PERSONALLY, CAN BE FOUND IN `compiler.py`*

This project implements anonymous functions *without* the use of Lambda.

Within compiler.py, there is only one new pass, 'convert-closures.'
In this pass, we do all of the work in the primary function, `lift_nested_functions`
This function renames and lifts any nested function definitions, as well as replacing their original definitions with the required closure objects.
Inside the fucntion, we call `free_vars_stmts`, which, much like the function structutre in RCO, takes a list of stmts, passes it to `free_vars_stmt,` then passes any expressions within that singular statement to `free_vars_exp.` 
This allows us to easily create closure objects, having a dedicated function for finding any free variables.

Originally, with `lift_nested_functions,` I struggled to properly implement recursivity. 
But, after some toying around with the code, I was able to get a psuedo-recursive version working.
This allows the nesting to be hypothetically unlimited, while also ensuring that the lifting never fails/misses a function.

No features that were planned were not implemented.
However, as mentioned before, Lambda was not implemented. 
If I were to spend more time on this project, that would be my next starting point.

<table>
  <tr>
    <th>Sample Input</th>
    <th>Sample Output</th>
  </tr>
  <tr>
  <td valign="top">
      
  ```python
  x = 5

  def f() -> int:
      return x

  x = 6

  print(f())
  ```

  </td>
  <td valign="top">
    
  ```assembly
    .globl main
  fstart:
    movq %r8, %rax
    jmp fconclusion
  f:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    subq $0, %rsp
    jmp fstart
  fconclusion:
    addq $0, %rsp
    subq $0, %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    retq
  mainstart:
    movq $5, %r14
    movq $6, %r14
    leaq f(%rip), %r13
    pushq %rdx
    pushq %rcx
    pushq %rsi
    pushq %rdi
    pushq %r8
    pushq %r9
    pushq %r10
    callq *%r13
    popq %r10
    popq %r9
    popq %r8
    popq %rdi
    popq %rsi
    popq %rcx
    popq %rdx
    movq %rax, %r8
    movq %r8, %rdi
    callq print_int
    movq $0, %rax
    jmp mainconclusion
  main:
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    subq $0, %rsp
    movq $16384, %rdi
    movq $16, %rsi
    callq initialize
    movq rootstack_begin(%rip), %r15
    jmp mainstart
  mainconclusion:
    addq $0, %rsp
    subq $0, %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    retq
  
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
  ```

  </td>
  </tr>
</table>
