import importlib

module = importlib.import_module("models")
class_type = getattr(module, "NaiveCloser")
print(class_type)
# print(globals())