import numpy
from stl import mesh

def create_stl_file(voxel_data):
    num_elements = 0
    for i in range(len(voxel_data)):
        for j in range(len(voxel_data[i])):
            num_elements += len(voxel_data[i][j])

    vertices = numpy.zeros((num_elements * 3 * 6, 3))
    faces = numpy.zeros((num_elements * 6, 3))
    i = 0
    j = 0
    # Iterate over each voxel in the data
    for z, layer in enumerate(voxel_data):
        for y, row in enumerate(layer):
            for x, voxel in enumerate(row):
                if voxel == 1:  # Filled voxel
                    # Calculate the vertices and normal for each face of the voxel

                    # Bottom face
                    v1 = (x, y, z)
                    v2 = (x + 1, y, z)
                    v3 = (x, y + 1, z)
                    v4 = (x + 1, y + 1, z)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i + 1, i + 2, i + 3]
                    j = j + 2
                    i = i + 4

                    # Top face
                    v1 = (x + 1, y + 1, z + 1)
                    v2 = (x, y + 1, z + 1)
                    v3 = (x + 1, y, z + 1)
                    v4 = (x, y, z + 1)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i + 1, i + 2, i + 3]
                    j = j + 2
                    i = i + 4

                    # Front face
                    v1 = (x, y, z)
                    v2 = (x, y + 1, z)
                    v3 = (x, y, z + 1)
                    v4 = (x, y + 1, z + 1)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i + 1, i + 2, i + 3]
                    j = j + 2
                    i = i + 4

                    # Back face
                    v1 = (x + 1, y, z + 1)
                    v2 = (x + 1, y + 1, z + 1)
                    v3 = (x + 1, y, z)
                    v4 = (x + 1, y + 1, z)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i, i + 2, i + 3]
                    j = j + 2
                    i = i + 4

                    # Left face
                    v1 = (x, y, z)
                    v2 = (x, y, z + 1)
                    v3 = (x + 1, y, z)
                    v4 = (x + 1, y, z + 1)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i + 1, i + 2, i + 3]
                    j = j + 2
                    i = i + 4

                    # Right face
                    v1 = (x + 1, y + 1, z)
                    v2 = (x + 1, y + 1, z + 1)
                    v3 = (x, y + 1, z)
                    v4 = (x + 1, y + 1, z + 1)
                    vertices[i] = v1
                    vertices[i + 1] = v2
                    vertices[i + 2] = v3
                    vertices[i + 3] = v4
                    faces[j] = [i, i+1, i+2]
                    faces[j + 1] = [i, i + 1, i + 3]
                    j = j + 2
                    i = i + 4
    return vertices, faces, i, j


def x_to_stl(nely, nelz, nelx, tol, output_filename):
    # Example usage
    filename = "x_opt.txt"  # Specify the name of your text file

    # Read the vector from the text file
    with open(filename, "r") as file:
        x_opt = [float(line.strip()) for line in file]

    # Assuming you have a 3D array 'design_domain' representing the discretized cubes
    # And a corresponding 3D array 'degree_of_saturation' representing the degree of saturation for each cube

    design_domain = numpy.reshape(x_opt,(nelz, nely, nelx))
    # Create meshgrid for the axes
    y, x, z = numpy.meshgrid(numpy.arange(design_domain.shape[0]),
                          numpy.arange(design_domain.shape[1]),
                          numpy.arange(design_domain.shape[2]))

    x = numpy.reshape(x, (len(x_opt)))
    y = numpy.reshape(y, (len(x_opt)))
    z = numpy.reshape(z, (len(x_opt)))

    x_opt_new = []
    x_new = []
    y_new = []
    z_new = []

    for i in range(len(x_opt)):
        if x_opt[i] >= tol:
            x_opt_new.append(x_opt[i])
            x_new.append(x[i])
            y_new.append(y[i])
            z_new.append(z[i])

    x = x_new
    y = y_new
    z = z_new

    x_length = numpy.max(x) + 1
    y_length = numpy.max(y) + 1
    z_length = numpy.max(z) + 1

    # # Create a binary array to represent the volume
    volume = numpy.zeros((x_length, y_length, z_length), dtype=bool)
    volume[x, y, z] = True

    vertices, faces, i, j = create_stl_file(volume)

    vertices = vertices[0:i, :].astype(int)
    faces = faces[0:j, :].astype(int)

    # Create the mesh
    cube = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]

    # Write the mesh to file "cube.stl"
    cube.save(output_filename)
