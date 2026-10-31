import numpy as np
import matplotlib.pyplot as plt

EPSILON = 1e-8

type point = np.array
type triangle = tuple[point, point, point]

def get_triangle_surface_cross(a: point, b: point, c: point):
    ab = (b - a)
    ac = (c - a)

    x1, y1 = ab
    x2, y2 = ac

    # Via cross product
    return 0.5 * np.abs(x1 * y2 - x2 * y1)

def get_triangle_surface_gorner(a: point, b: point, c: point):
    ab = np.linalg.norm(b - a)
    ac = np.linalg.norm(c - a)
    bc = np.linalg.norm(c - b)

    p = (ab + ac + bc) / 2

    return np.sqrt(p * (p - ab) * (p - ac) * (p - bc))

def distance_point_to_segment(a: point, b: point, c: point):
    ab = np.linalg.norm(b - a)
    ac = np.linalg.norm(c - a)
    bc = np.linalg.norm(c - b)

    cos_angle = (ac**2 + ab**2 - bc**2) / (2 * ac * ab)
    sin_angle = np.sqrt(max(0, 1 - cos_angle**2))

    # via surface
    area = 0.5 * ac * ab * sin_angle
    height = 2 * area / ab

    return height

def is_point_in_triangle(triangle: tuple[point, point, point], p: point):
    a, b, c = triangle

    S = get_triangle_surface_cross(a, b, c)
    S_through_p = get_triangle_surface_cross(a, b, p) + get_triangle_surface_cross(a, c, p) + get_triangle_surface_cross(b, c, p)

    return np.abs(S - S_through_p) < EPSILON


class RegularPolygon:
    def __init__(self, n, radius):
        assert n >= 3
        assert radius > 0

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        self.vertices = [
            np.array([radius * np.cos(a), radius * np.sin(a)])
            for a in angles
        ]

        self.sides = [
            (self.vertices[i], self.vertices[(i + 1) % n])
            for i in range(n)
        ]

        self.diagonals = [
            (self.vertices[i], self.vertices[j])
            for i in range(n)
            for j in range(i + 1, n)
            if j != i + 1 and not (i == 0 and j == n - 1)
        ]

        self.n = n
        self.radius = radius

    def draw(self):
        fig, ax = plt.subplots()

        for a, b in self.diagonals:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='green', linewidth=0.8)

        for a, b in self.sides:
            ax.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=1.5)

        vertices = np.array(self.vertices)
        ax.scatter(vertices[:, 0], vertices[:, 1], color='red', s=40, zorder=3)

        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()

    def closest_distance_from_side(self, point):
        return min(
            distance_point_to_segment(a, b, point)
            for a, b in self.sides
        )

    def closest_distance_from_diagonal(self, point):
        if not self.diagonals:
            return np.nan

        return min(
            distance_point_to_segment(a, b, point)
            for a, b in self.diagonals
        )

    def is_inner(self, point: point):
        center = np.array((0, 0))

        # iterate through all the triangles that
        # this regular polygon is built of
        for i in range(self.n):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % self.n]

            if is_point_in_triangle((a, b, center), point):
                return True

        return False

    def random_inner_point(self, n_max_iter=100) -> point:
        # rejection sampling

        for i in range(n_max_iter):
            xy = np.random.uniform(-self.radius, self.radius, size=2)

            if xy.dot(xy) > self.radius * self.radius:
                continue

            if not self.is_inner(xy):
                continue

            return xy

        raise Exception(f"Could not generate inner point in {n_max_iter} iterations")

