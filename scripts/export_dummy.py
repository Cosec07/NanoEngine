import struct

data = [1.0, 2.0, 3.0, 4.0]

with open("model.bin", "wb") as f:
    binary_data = struct.pack(f'{len(data)}f', *data)
    f.write(binary_data)

print(f"Exported {len(data)} floats to model.bin")