import h5py

# Load
with h5py.File('data.h5', 'r') as hf:
    data = hf['data'][:]
    metadata = eval(hf.attrs['metadata'])

print(data)
print(metadata)
