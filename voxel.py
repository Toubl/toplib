import numpy
from stl import mesh

def create_stl_file(voxel_data):
    num_elements = 0
    for i in range(len(voxel_data)):
        for j in range(len(voxel_data[i])):
            num_elements += len(voxel_data[i][j])

    vertices = numpy.zeros((num_elements * 3 * 6 * 2, 3))
    faces = numpy.zeros((num_elements * 6 * 2, 3))
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


def x_to_stl(nelx, nely, nelz, tol, x, output_filename):

    volume = numpy.reshape(x, (nelx, nelz, nely), order='C') > tol


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



# import trimesh
#
# # Load the STL file
# mesh2 = trimesh.load_mesh('output.stl')
#
# # Smooth the mesh using Humphrey smoothing
# mesh2 = trimesh.smoothing.filter_humphrey(mesh2, alpha=1, beta=0, iterations=100)
# mesh2 = mesh2.subdivide()
# mesh2 = trimesh.smoothing.filter_humphrey(mesh2, alpha=1, beta=0, iterations=100)
#
# # Save the smoothed mesh as an STL file
# mesh2.export('smooth.stl')

    import numpy as np
    from pyvista import CellType
    import pyvista

    # Define the coordinate ranges
    x_range = np.linspace(0, volume.shape[0], num=volume.shape[0] + 1)
    y_range = np.linspace(0, volume.shape[1], num=volume.shape[1] + 1)
    z_range = np.linspace(0, volume.shape[2], num=volume.shape[2] + 1)

    # Create the meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))
    Z = np.reshape(Z, (-1, 1))
    points = np.stack((X, Y, Z), axis=1)
    points = np.squeeze(points)
    indices = np.arange((volume.shape[0] + 1) * (volume.shape[1] + 1) * (volume.shape[2] + 1))
    indices = np.reshape(indices, (volume.shape[0] + 1, volume.shape[1] + 1, volume.shape[2] + 1))

    m = 0
    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                if volume[x, y, z]:
                    cell = np.array([8, indices[x, y, z], indices[x + 1, y, z], indices[x + 1, y + 1, z], indices[x, y + 1, z],
                                     indices[x, y, z + 1], indices[x + 1, y, z + 1], indices[x + 1, y + 1, z + 1], indices[x, y + 1, z + 1]])

                    cell_type = np.array([CellType.HEXAHEDRON], np.int8)
                    if m == 0:
                        cell_types = cell_type
                        cells = cell
                    else:
                        cell_types = np.concatenate((cell_types, cell_type))
                        cells = np.concatenate((cells, cell))
                    m += 8

    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    surf = grid.extract_geometry()


    smooth_w_taubin = surf.smooth_taubin(n_iter=50, pass_band=0.1)
    surf.plot(show_edges=True, show_scalar_bar=False)
    smooth_w_taubin.plot(show_edges=True, show_scalar_bar=False)

    filename = "smooth.stl"
    smooth_w_taubin.save(filename)

#
# x_opt = []
# with open('x_opt.txt', 'r') as file:
#     for line in file:
#         entry = line.strip()
#         x_opt.append(entry)
#
# x_opt = numpy.array(x_opt).astype(numpy.float64)
# x_to_stl(40, 20, 20, 0.1, x_opt, 'smooth.stl')