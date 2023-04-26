import pyfiglet

title = pyfiglet.figlet_format("TopOpt", font="doh", width=200)
with open('utils/ascii_ose.txt') as f:
    ose=f.read()

print(title)
print(ose)
