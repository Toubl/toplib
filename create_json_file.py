import json

nelx, nely = 200, 70  # Number of elements in the x and y
volfrac = 0.36  # Volume fraction for constraints
penal = 3.0  # Penalty for SIMP
rmin = 4.4  # Filter radius

data = {
    "nelx": nelx,
    "nely": nely,
    "volfrac": volfrac,
    "penal": penal,
    "rmin": rmin
}

filename = "save_2.json"

with open(filename, "x") as outfile:
    json.dump(data, outfile)

print(f"JSON file '{filename}' created successfully.")
