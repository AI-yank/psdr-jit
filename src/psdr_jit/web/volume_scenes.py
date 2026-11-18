"""Self-contained homogeneous scenes for the demo and reference renders."""

VOLUME_BOUNDS = ((-.75, -.75, 1.), (.75, .75, 2.))
FOG_CUBE_BOUNDS = ((-.75, -.75, -.75), (.75, .75, .75))


def _scene_with_camera(size, spp, camera):
    import psdr_jit as psdr

    scene = psdr.Scene()
    scene.opts = psdr.RenderOption(size, size, spp, 0, 0)
    scene.opts.log_level = 0
    scene.add_Sensor(camera)
    scene.add_BSDF(psdr.DiffuseBSDF([0, 0, 0]), "black")
    return scene


def volume_scene(size, spp, light=1.):
    """Backlit fog with an orthographic camera and a rectangular area emitter."""
    import psdr_jit as psdr
    from drjit.cuda import Array3f, Array3i

    scene = _scene_with_camera(size, spp, psdr.OrthographicCamera(.01, 100.))
    panel = psdr.Mesh()
    panel.load_raw(Array3f([[-.25, .25, .25, -.25], [-.5, -.5, .5, .5], [2.5] * 4]),
                   Array3i([[0, 0], [2, 3], [1, 2]]))
    scene.add_Mesh(panel, "black", psdr.AreaLight([4 * light, 2 * light, light]))
    scene.configure()
    return scene


def fog_cube_scene(size, spp, light=1.):
    """View a free-standing cube of fog from above, between two colored area lights."""
    import numpy as np
    import psdr_jit as psdr
    from drjit.cuda import Array3f, Array3i
    from drjit.cuda.ad import Matrix4f

    camera = psdr.PerspectiveCamera(38., .01, 100.)
    eye = np.array([2.4, 1.85, -3.3])
    forward = -eye / np.linalg.norm(eye)
    horizontal = np.cross([0., 1., 0.], forward)
    horizontal /= np.linalg.norm(horizontal)
    transform = np.eye(4)
    transform[:3] = np.column_stack((horizontal, np.cross(forward, horizontal), forward, eye))
    camera.to_world = Matrix4f(transform.tolist())
    sc = _scene_with_camera(size, spp, camera)

    # Aim the off-camera emitters toward the cube. Its visible silhouette
    # comes entirely from light scattered along the camera rays.
    for center, color in [([1.9, 1.0, 1.1], [13., 4.6, 1.1]),
                          ([-1.9, .1, -1.1], [.8, 5.4, 14.])]:
        center = np.array(center)
        normal = -center / np.linalg.norm(center)
        u = np.cross([0., 1., 0.], normal)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u) * .9
        u *= .65
        points = np.stack([center - u - v, center + u - v, center + u + v, center - u + v])
        panel = psdr.Mesh()
        panel.load_raw(Array3f(points.T.copy()), Array3i([[0, 0], [1, 2], [2, 3]]))
        sc.add_Mesh(panel, "black", psdr.AreaLight([c * light for c in color]))
    # A thin luminous grid beneath the medium makes transmission visible.
    # Build disjoint strips so intersections do not double the emission.
    strips = []
    ticks = np.linspace(-.88, .88, 5)
    half = .007
    for x in ticks:
        strips.append((x - half, x + half, ticks[0] - half, ticks[-1] + half))
    for left, right in zip(ticks, ticks[1:]):
        for z in ticks:
            strips.append((left + half, right - half, z - half, z + half))
    grid_points, triangles = [], []
    for x0, x1, z0, z1 in strips:
        i = len(grid_points)
        grid_points.extend([(x0, -.82, z0), (x0, -.82, z1), (x1, -.82, z1), (x1, -.82, z0)])
        triangles.extend([(i, i + 1, i + 2), (i, i + 2, i + 3)])
    grid = psdr.Mesh()
    grid.load_raw(Array3f(np.array(grid_points).T.copy()), Array3i(np.array(triangles).T.copy()))
    sc.add_Mesh(grid, "black", psdr.AreaLight([.65 * light] * 3))
    sc.configure()
    return sc
