z = 1
def f() -> int:
    z = 3
    def g() -> int:
        print(z)
        return 0
    def h() -> int:
        print(z)
        return 0
    return g()

z = 5
print(f())
