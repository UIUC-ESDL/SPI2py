import pyvista as pv

# Read each line in files/part2_mdbd.txt and convert it to a list of floats

with open('files/part2_mdbd.txt', 'r') as f:
    lines = f.readlines()

p = pv.Plotter()

lines = lines[0:150]

for line in lines:
    x,y,z,r = line.split()

    center = [float(x), float(y), float(z)]
    radius = float(r)

    sphere = pv.Sphere(center=center, radius=radius)
    p.add_mesh(sphere, color='tan', opacity=0.5)

p.show()