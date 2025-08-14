import ast

loost = [1, 2, 3, 4, 5]
with open("./test.txt", 'w') as file:
    file.write(str(loost))


# with open("./test.txt", 'r') as file:
#     red = file.read()
#     red_list = ast.literal_eval(red)
#     print(red)
#     # print(list(red))
#     print(red_list)
#     print(type(red_list))
#     print(red_list[2])