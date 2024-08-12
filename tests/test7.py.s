  .globl main
h_funstart:
  movq %rdi, %r8
  movq %r8, %r11
  movq 8(%r11), %r8
  movq %r8, %rdi
  callq print_int
  movq $0, %rax
  jmp h_funconclusion
h_fun:
  pushq %rbp
  movq %rsp, %rbp
  pushq %rbx
  pushq %r12
  pushq %r13
  pushq %r14
  subq $0, %rsp
  jmp h_funstart
h_funconclusion:
  addq $0, %rsp
  subq $0, %r15
  popq %r14
  popq %r13
  popq %r12
  popq %rbx
  popq %rbp
  retq
fstart:
  movq $3, %r13
  movq $24, %rdi
  callq allocate
  movq %rax, %r11
  movq $3, 0(%r11)
  movq %r14, 8(%r11)
  movq %r13, 16(%r11)
  movq %r11, -8(%r15)
  movq $24, %rdi
  callq allocate
  movq %rax, %r11
  movq $3, 0(%r11)
  movq %r14, 8(%r11)
  movq %r13, 16(%r11)
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
