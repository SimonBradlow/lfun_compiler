z = 1
def f() -> int:
    z = 3
    def g() -> int:
        print(z)
        return z
    return g()

def h() -> int:
    z = 7
    def i() -> int:
        print(z)
        return z
    return i()

z = 5
print(f())
