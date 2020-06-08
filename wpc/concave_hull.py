# from shapely.ops import cascaded_union, polygonize
# from scipy.spatial import Delaunay
# import numpy as np
# import shapely.geometry as geometry
# import math
#
#
# def alpha_shape(points, alpha):
#     """
#     Compute the alpha shape (concave hull) of a set
#     of points.
#     @param points: Iterable container of points.
#     @param alpha: alpha value to influence the
#         gooeyness of the border. Smaller numbers
#         don't fall inward as much as larger numbers.
#         Too large, and you lose everything!
#     """
#     if len(points) < 4:
#         # When you have a triangle, there is no sense
#         # in computing an alpha shape.
#         return geometry.MultiPoint(list(points)).convex_hull
#
#     def add_edge(edges, edge_points, coords, i, j):
#         """
#         Add a line between the i-th and j-th points,
#         if not in the list already
#         """
#         if (i, j) in edges or (j, i) in edges:
#             # already added
#             return
#         edges.add( (i, j) )
#         edge_points.append(coords[ [i, j] ])
#
#     coords = np.array([point.coords[0]
#                        for point in points])
#     tri = Delaunay(coords)
#     edges = set()
#     edge_points = []
#     # loop over triangles:
#     # ia, ib, ic = indices of corner points of the
#     # triangle
#     for ia, ib, ic in tri.vertices:
#         pa = coords[ia]
#         pb = coords[ib]
#         pc = coords[ic]
#         # Lengths of sides of triangle
#         a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
#         b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
#         c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
#         # Semiperimeter of triangle
#         s = (a + b + c)/2.0
#         # Area of triangle by Heron's formula
#         area = math.sqrt(s*(s-a)*(s-b)*(s-c))
#         circum_r = a*b*c/(4.0*area)
#         # Here's the radius filter.
#         #print circum_r
#         if circum_r < 1.0/alpha:
#             add_edge(edges, edge_points, coords, ia, ib)
#             add_edge(edges, edge_points, coords, ib, ic)
#             add_edge(edges, edge_points, coords, ic, ia)
#     m = geometry.MultiLineString(edge_points)
#     triangles = list(polygonize(m))
#     return cascaded_union(triangles), edge_points


import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()