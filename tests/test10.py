x = 5

def f() -> int:
    if x == 6:
        print(6)
    else:
        print(5)
    return x

x = 6

print(f())
