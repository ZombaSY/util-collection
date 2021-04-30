my_str = 'sunyongfactoryhihifactoryzzzzzfactory'
temp = my_str.split('factory')
for item in temp:
    if len(item) > 0:
        print(item)