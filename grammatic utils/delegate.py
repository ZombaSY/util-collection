
def plus(a, b):
    return a + b


def minus(a, b):
    return a - b


def cal(a, b, arg):
    return arg(a, b)


print(cal(10, 5, minus))

