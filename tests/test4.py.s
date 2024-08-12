  .globl main
i_funstart:
  movq %rdi, %r8
  movq %r8, %r11
  movq 8(%r11), %r8
  movq %r8, %rdi
  callq print_int
  movq %r8, %rax
  jmp i_funconclusion
i_fun:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  subq $0, %rsp
  jmp i_funstart
i_funconclusion:
  addq $0, %rsp
  subq $0, %r15
  popq %r14
  popq %r13
  popq %r12
  popq %rbx
  popq %rbp
  retq
h_funstart:
  movq %rdi, %r14
  movq %r14, %r11
  movq 8(%r11), %r13
  movq %r14, %r11
  movq 8(%r11), %rax
  movq %rax, -8(%r15)
  movq %r14, %r11
  movq 8(%r11), %r14
  movq %r13, %rdi
  callq print_int
  movq $24, %rdi
  callq allocate
  movq %rax, %r11
  movq $3, 0(%r11)
  movq %r12, 8(%r11)
  movq %r13, 16(%r11)
  movq %r11, -8(%r15)
  leaq i(%rip), %r12
  pushq %rdx
  pushq %rcx
  pushq %rsi
  pushq %rdi
  pushq %r8
  pushq %r9
  pushq %r10
  callq *%r12
  popq %r10
  popq %r9
  popq %r8
  popq %rdi
  popq %rsi
  popq %rcx
  popq %rdx
  movq %rax, %r14
  movq %r14, %rax
  jmp h_funconclusion
h_fun:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  subq $0, %rsp
  movq $0, 0(%r15)
  addq $8, %r15
  jmp h_funstart
h_funconclusion:
  addq $0, %rsp
  subq $8, %r15
  popq %r14
  popq %r13
  popq %r12
  popq %rbx
  popq %rbp
  retq
g_funstart:
  movq %rdi, %r13
  movq %r13, %r11
  movq 8(%r11), %r12
  movq %r13, %r11
  movq 8(%r11), %rax
  movq %rax, -8(%r15)
  movq %r13, %r11
  movq 8(%r11), %r13
  movq %r12, %rdi
  callq print_int
  movq $40, %rdi
  callq allocate
  movq %rax, %r11
  movq $7, 0(%r11)
  movq %r14, 8(%r11)
  movq %r12, 16(%r11)
  movq -16(%r15), %rax
  movq %rax, 24(%r11)
  movq %r14, 32(%r11)
  movq %r11, -8(%r15)
  leaq h(%rip), %r14
  pushq %rdx
  pushq %rcx
  pushq %rsi
  pushq %rdi
  pushq %r8
  pushq %r9
  pushq %r10
  callq *%r14
  popq %r10
  popq %r9
  popq %r8
  popq %rdi
  popq %rsi
  popq %rcx
  popq %rdx
  movq %rax, %r13
  movq %r13, %rax
  jmp g_funconclusion
g_fun:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  subq $0, %rsp
  movq $0, 0(%r15)
  addq $8, %r15
  movq $0, 0(%r15)
  addq $8, %r15
  jmp g_funstart
g_funconclusion:
  addq $0, %rsp
  subq $16, %r15
  popq %r14
  popq %r13
  popq %r12
  popq %rbx
  popq %rbp
  retq
fstart:
  movq $3, %r13
  movq $40, %rdi
  callq allocate
  movq %rax, %r11
  movq $7, 0(%r11)
  movq %r14, 8(%r11)
  movq %r13, 16(%r11)
  movq -8(%r15), %rax
  movq %rax, 24(%r11)
  movq %r14, 32(%r11)
  movq %r11, -16(%r15)
  leaq g(%rip), %r14
  pushq %rdx
  pushq %rcx
  pushq %rsi
  pushq %rdi
  pushq %r8
  pushq %r9
  pushq %r10
  callq *%r14
  popq %r10
  popq %r9
  popq %r8
  popq %rdi
  popq %rsi
  popq %rcx
  popq %rdx
  movq %rax, %r8
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
  movq $0, 0(%r15)
  addq $8, %r15
  movq $0, 0(%r15)
  addq $8, %r15
  jmp fstart
fconclusion:
  addq $0, %rsp
  subq $16, %r15
  popq %r14
  popq %r13
  popq %r12
  popq %rbx
  popq %rbp
  retq
mainstart:
  movq $1, %r13
  movq $5, %r13
  leaq f(%rip), %r14
  pushq %rdx
  pushq %rcx
  pushq %rsi
  pushq %rdi
  pushq %r8
  pushq %r9
  pushq %r10
  callq *%r14
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
