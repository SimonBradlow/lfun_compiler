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
