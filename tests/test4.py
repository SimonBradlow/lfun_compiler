z = 1
def f() -> int:
    z = 3
    def g() -> int:
        print(z)
        def h() -> int:
            print(z)
            def i() -> int:
                print(z)
                return z
            return i()
        return h()
    return g()

z = 5
print(f())
