'''
This code traces streamlines on Gaussian blobs. Non-Morse regions are filtered.
Author: Congrong Ren
Date: Nov 20, 2023
'''
import numpy as np
import pickle
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay
from scipy.stats.qmc import PoissonDisk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

x_dim = 400
y_dim = 400
x_range = [-20, 20]
y_range = [-20, 20]
round_num_seeds = 30
np.random.seed(2314)
avg_num_samples = 2000
time_interval = 100
degenerate_threshold_gradient = 5e-2
degenerate_threshold_eigvals = 1e-2


def circle(pts):
    a = pts[0][0] - pts[1][0]
    b = pts[0][1] - pts[1][1]
    c = pts[0][0] - pts[2][0]
    d = pts[0][1] - pts[2][1]
    a1 = (pts[0][0] ** 2 - pts[1][0] ** 2 + pts[0][1] ** 2 - pts[1][1] ** 2) / 2
    a2 = (pts[0][0] ** 2 - pts[2][0] ** 2 + pts[0][1] ** 2 - pts[2][1] ** 2) / 2
    theta = b * c - a * d
    if np.abs(theta) < 1e-7:
        raise RuntimeError('Input three different points!')
    x = (b * a2 - d * a1) / theta
    y = (c * a1 - a * a2) / theta
    return np.array([x, y])


def trisPlot2D(cells, cell_nodes_coor, streamlines, criticalPts, criticalPts_types, x_space, y_space, field,
               Morse_region=None, fig_id=0):
    ax = plt.figure().add_subplot()
    ax.pcolormesh(x_space, y_space, field, cmap='Greys')
    ax.triplot(cell_nodes_coor[:, 0], cell_nodes_coor[:, 1], cells)
    # for line in streamlines:
    #     ax.plot(line[:, 0], line[:, 1], color='g', alpha=0.3)
    ax.scatter(cell_nodes_coor[:, 0], cell_nodes_coor[:, 1], c='b', s=3)
    colormap = {1: 'g', 2: 'r', 3: "y"}
    criticalPts_colors = [colormap[t] for t in criticalPts_types]
    ax.scatter(criticalPts[:, 0], criticalPts[:, 1], c=criticalPts_colors)
    ax.scatter(4.2, 1.3, c='pink', s=2)
    if Morse_region is not None:
        plt.pcolormesh(x_space, y_space, Morse_region, cmap='Greens', alpha=0.1)
    plt.show()
    # plt.savefig("adaptive_fig/center/" + str(fig_id) + ".png")


def trisPlotColor(cells, cells_labels, cell_nodes_coor, streamlines, criticalPts, criticalPts_types, x_space, y_space,
                  field, Morse_region=None, fig_id=0):
    ax = plt.figure().add_subplot()
    ax.pcolormesh(x_space, y_space, field, cmap='Greys')
    ax.triplot(cell_nodes_coor[:, 0], cell_nodes_coor[:, 1], cells)
    ax.scatter(cell_nodes_coor[:, 0], cell_nodes_coor[:, 1], c='b', s=3)
    colormap = {1: 'g', 2: 'r', 3: "y"}
    criticalPts_colors = [colormap[t] for t in criticalPts_types]
    ax.scatter(criticalPts[:, 0], criticalPts[:, 1], c=criticalPts_colors)
    ax.scatter(4.2, 1.3, c='pink', s=2)
    if Morse_region is not None:
        plt.pcolormesh(x_space, y_space, Morse_region, cmap='Greens', alpha=0.1)
    for i, cl in enumerate(cells_labels):
        if cl == 1:
            pts = cell_nodes_coor[cells[i]]
            plt.plot((pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0]), (pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1]),
                     color='blue')
            plt.fill((pts[0, 0], pts[1, 0], pts[2, 0], pts[0, 0]), (pts[0, 1], pts[1, 1], pts[2, 1], pts[0, 1]),
                     color='#ff33b5')
    # plt.show()
    plt.savefig("adaptive_fig/center/" + str(fig_id) + ".png")


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def twoDGaussian(pts: np.array, param):
    '''
    Args:
    pts: shape = (num_pts, 2)
    param: [mu_x, mu_y, sigma_x, sigma_y, correlation]
    Return: values at pts
    '''
    norms_x = (pts[:, 0] - param[0]) / param[2]
    norms_y = (pts[:, 1] - param[1]) / param[3]
    return np.exp(-0.5 * (norms_x ** 2 - 2 * param[4] * norms_x * norms_y + norms_y ** 2) / (1 - param[4] ** 2)) / (
            2 * np.pi * param[2] * param[3] * np.sqrt(1 - param[4] ** 2))


def twoDGaussian_gradients(pts: np.array, param):
    '''
    Args:
    pts: shape = (num_pts, 2)
    param: [mu_x, mu_y, sigma_x, sigma_y, correlation]
    Return: gradientss at pts
    '''
    num_pts = pts.shape[0]
    norms_x = (pts[:, 0] - param[0]) / param[2]
    norms_y = (pts[:, 1] - param[1]) / param[3]
    Gaussian = twoDGaussian(pts, param)
    grads = np.zeros((num_pts, 2))
    grads[:, 0] = Gaussian * (param[4] * norms_y - norms_x) / (param[2] * (1 - param[4] ** 2))
    grads[:, 1] = Gaussian * (param[4] * norms_x - norms_y) / (param[3] * (1 - param[4] ** 2))
    return grads


def twoDGaussian_Hessian(pts: np.array, param):
    '''
        Args:
        pts: shape = (num_pts, 2)
        param: [mu_x, mu_y, sigma_x, sigma_y, correlation]
        Return: Hessian at pts
    '''
    num_pts = pts.shape[0]
    norms_x = (pts[:, 0] - param[0]) / param[2]
    norms_y = (pts[:, 1] - param[1]) / param[3]
    Gaussian = twoDGaussian(pts, param)
    Hessian = np.zeros((num_pts, 2, 2))
    Hessian[:, 0, 0] = Gaussian * ((norms_x - param[4] * norms_y) ** 2 / (1 - param[4] ** 2) - 1) / (
            param[2] ** 2 * (1 - param[4] ** 2))
    Hessian[:, 0, 1] = Gaussian * (
            (norms_x - param[4] * norms_y) * (norms_y - param[4] * norms_x) / (1 - param[4] ** 2) + param[4]) / (
                               param[2] * param[3] * (1 - param[4] ** 2))
    Hessian[:, 1, 0] = Hessian[:, 0, 1]
    Hessian[:, 1, 1] = Gaussian * ((norms_y - param[4] * norms_x) ** 2 / (1 - param[4] ** 2) - 1) / (
            param[3] ** 2 * (1 - param[4] ** 2))
    return Hessian


def mixtureOf2dGaussian(pts: np.array, params):
    '''
        Args:
        pts: shape = (num_pts, 2)
        params: a list of parameters [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
        Return: values at pts
    '''
    num_points = pts.shape[0]
    scalar_value = np.zeros(num_points)
    for param in params:
        scalar_value += param[-1] * twoDGaussian(pts, param[:-1])
    return scalar_value


def mixtureOf2dGaussianGradient(pts: np.array, params):
    '''
        Args:
        pts: shape = (num_pts, 2)
        params: a list of parameters [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
        Return: gradients at pts
    '''
    num_points = pts.shape[0]
    grads = np.zeros((num_points, 2))
    for param in params:
        grads += param[-1] * twoDGaussian_gradients(pts, param[:-1])
    return grads


def mixtureOf2dGaussianHessian(pts: np.array, params):
    '''
        Args:
        pts: shape = (num_pts, 2)
        params: a list of parameters [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
        Return: Hessian at pts
    '''
    num_points = pts.shape[0]
    Hessian = np.zeros((num_points, 2, 2))
    for param in params:
        Hessian += param[-1] * twoDGaussian_Hessian(pts, param[:-1])
    return Hessian


def mixtureOf2dGaussianField(params, x, y, x_dim, y_dim):
    '''
    Args:
    params: a list of lists [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
    x: np.array in shape = (x_dim, y_dim)
    y: np.array in shape = (x_dim, y_dim)
    x_dim: number of pieces into which x-axis is divided
    y_dim: number of pieces into which y-axis is divided
    Return: solution of a scalar field
    '''
    pts = np.array([x, y]).reshape(2, -1).T
    scalar_value = mixtureOf2dGaussian(pts, params)
    scalar_value = scalar_value.reshape(x_dim, y_dim)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x[0], y[:, 0], scalar_value, cmap='Greys')
    # plt.show()
    return scalar_value


def mixtureOf2dGaussianFieldGradient(params, x, y, x_dim, y_dim):
    '''
    Args:
    params: a list of lists [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
    x: np.array in shape = (x_dim, y_dim)
    y: np.array in shape = (x_dim, y_dim)
    x_dim: number of pieces into which x-axis is divided
    y_dim: number of pieces into which y-axis is divided
    Return: gradient of a scalar field
    '''
    pts = np.array([x, y]).reshape(2, -1).T
    grads = mixtureOf2dGaussianGradient(pts, params)
    grads = grads.reshape(x_dim, y_dim, 2)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x[0], y[:, 0], grads[:, :, 0], cmap='Greys')
    # plt.show()
    return grads


def mixtureOf2dGaussianFieldHessian(params, x, y, x_dim, y_dim):
    '''
    Args:
    params: a list of lists [mu_x, mu_y, sigma_x, sigma_y, correlation, magnitude]
    x: np.array in shape = (x_dim, y_dim)
    y: np.array in shape = (x_dim, y_dim)
    x_dim: number of pieces into which x-axis is divided
    y_dim: number of pieces into which y-axis is divided
    Return: Hessian of a scalar field
    '''
    pts = np.array([x, y]).reshape(2, -1).T
    Hessian = mixtureOf2dGaussianHessian(pts, params)
    Hessian = Hessian.reshape(x_dim, y_dim, 2, 2)
    # plt.figure(figsize=(6, 6))
    # plt.pcolormesh(x[0], y[:, 0], np.log(np.abs(np.linalg.det(Hessian))), cmap='jet')
    # plt.show()
    return Hessian


def streamline_trace(point_coor, params, t_span, num_steps):
    '''
    Args:
        t: time duration in which the point moves
        num_steps: number of integration steps

    Returns: point's coordinate after delta_t
    '''

    num_half_steps = num_steps // 2
    dt = t_span / num_half_steps
    t_steps = (np.arange(num_half_steps) + 1) * dt

    def dP(t, point_coor):
        return mixtureOf2dGaussianGradient(np.array([point_coor]), params)[0]

    sol_forward = solve_ivp(dP, t_span=(0, t_span), y0=point_coor, method='RK45', t_eval=t_steps)
    sol_backward = solve_ivp(dP, t_span=(0, -t_span), y0=point_coor, method='RK45', t_eval=-t_steps)
    streamline = np.zeros((num_steps + 1, 2))
    streamline[:num_half_steps] = sol_forward.y.T
    streamline[num_half_steps] = point_coor
    streamline[-num_half_steps:] = sol_backward.y.T

    return streamline


def criticalPointsFromStreamlines(streamlines: np.array, params, t, num_steps):
    '''
    :param streamlines:
    :return: critical points and their types
    1 - min
    2 - max
    3 -  saddle
    '''
    dt = t / (num_steps // 2)
    threshold_large_dist = 1e-3 * dt
    threshold_small_dist = 1e-4 * dt
    threshold_critical_dist = 1e-3 * dt
    num_streamlines = streamlines.shape[0]
    delta_dist = np.linalg.norm(streamlines[:, 1:] - streamlines[:, :-1], axis=-1)
    change_steps = np.where(delta_dist > threshold_large_dist)
    fixed_steps = np.where(delta_dist < threshold_small_dist)
    # find streamlines trapped by critical points / filter out streamlines that do not either move or converge
    trapped_streamlines = list(set(change_steps[0]).intersection(set(fixed_steps[0])))
    num_criticalPts_temp = len(trapped_streamlines)
    criticalPts_temp = []
    mesh_nodes_coors = []
    streamline_lens = []
    curr_critical_id = 0
    for i in range(num_streamlines):
        trapped_pts_id = np.where(fixed_steps[0] == i)[0]
        trapped_pts_id = fixed_steps[1][trapped_pts_id]
        # print(trapped_pts_id)
        gaps = [[s, e] for s, e in zip(trapped_pts_id[:-1], trapped_pts_id[1:]) if s + 1 < e]
        trapped_pts_id_list = list(trapped_pts_id)
        edges = iter(list(trapped_pts_id_list[:1]) + sum(gaps, []) + list(trapped_pts_id_list[-1:]))
        trapped_pts_range = list(zip(edges, edges))
        for prev_range, curr_range in zip(trapped_pts_range[:-1], trapped_pts_range[1:]):
            for j in range(prev_range[1] + 1, curr_range[0]):
                mesh_nodes_coors.append(streamlines[i, j])
            mesh_nodes_coors.append(np.average(streamlines[i, curr_range[0]:curr_range[1] + 1], axis=0))
            streamline_lens.append(curr_range[0] - prev_range[1])
        if curr_critical_id < num_criticalPts_temp:
            if i == trapped_streamlines[curr_critical_id]:
                critical_pt_temp = streamlines[i, trapped_pts_id]
                criticalPts_temp.append(np.average(critical_pt_temp, axis=0))
                curr_critical_id += 1

    criticalPts = []
    criticalPts_types = []
    for i in range(num_criticalPts_temp):
        pt_temp = criticalPts_temp[i]
        repeat = False
        for pt in criticalPts:
            dist = np.linalg.norm(pt_temp - pt)
            if dist < threshold_critical_dist:
                repeat = True
                break
        if not repeat:
            criticalPts.append(pt_temp)
            Hessian = mixtureOf2dGaussianHessian(np.array([pt_temp]), params)[0]
            eig_vals, eig_vecs = np.linalg.eig(Hessian)
            if all(eig_vals > 0):
                criticalPts_types.append(1)
            elif all(eig_vals < 0):
                criticalPts_types.append(2)
            else:
                criticalPts_types.append(3)

    curr_num_nodes = 0
    terminate_criticalPts = []
    for i in range(num_criticalPts_temp):
        curr_num_nodes += streamline_lens[i]
        curr_terminate_criticalPt = mesh_nodes_coors[curr_num_nodes - 1]
        for j, ct in enumerate(criticalPts):
            if np.linalg.norm(curr_terminate_criticalPt - ct) < threshold_critical_dist:
                terminate_criticalPts += [j] * streamline_lens[i]
                break

    return np.array(criticalPts), criticalPts_types, np.array(mesh_nodes_coors), terminate_criticalPts


def updateCriticalPointsFromOneStreamline(streamline: np.array, criticalPts: np.array, criticalPts_types: list,
                                          num_cps: int, mesh_nodes: np.array, mesh_nodes_labels: list,
                                          num_mesh_nodes: int, params, t, num_steps):
    dt = t / (num_steps // 2)
    threshold_large_dist = 5e-3 * dt
    threshold_small_dist = 1e-5 * dt
    threshold_critical_dist = 1e-3 * dt
    delta_dist = np.linalg.norm(streamline[1:] - streamline[:-1], axis=-1)
    change_steps = np.where(delta_dist > threshold_large_dist)[0]
    fixed_steps = np.where(delta_dist < threshold_small_dist)[0]
    # find streamlines trapped by critical points / filter out streamlines that do not either move or converge
    if_trapped = change_steps.shape[0] > 0 and fixed_steps.shape[0] > 0
    if if_trapped:
        num_added_nodes = change_steps.shape[0] + 1
        mesh_nodes[num_mesh_nodes:num_mesh_nodes + num_added_nodes - 1] = streamline[change_steps]
        num_mesh_nodes += num_added_nodes
        mesh_nodes[num_mesh_nodes] = np.mean(streamline[fixed_steps], axis=0)
        for i, ct in enumerate(criticalPts):
            if np.linalg.norm(mesh_nodes[num_mesh_nodes] - ct) < 0.1:
                mesh_nodes_labels += [i] * num_added_nodes
                break
    print(num_mesh_nodes)

    # if curr_critical_id < num_criticalPts_temp:
    #     if i == trapped_streamlines[curr_critical_id]:
    #         critical_pt_temp = streamlines[i, fixed_steps[1][trapped_pts_id[-2:]]]
    #         criticalPts_temp.append(np.average(critical_pt_temp, axis=0))
    #         curr_critical_id += 1
    # criticalPts = []
    # criticalPts_types = []
    # for i in range(num_criticalPts_temp):
    #     pt_temp = criticalPts_temp[i]
    #     repeat = False
    #     for pt in criticalPts:
    #         dist = np.linalg.norm(pt_temp - pt)
    #         if dist < threshold_critical_dist:
    #             repeat = True
    #             break
    #     if not repeat:
    #         criticalPts.append(pt_temp)
    #         Hessian = mixtureOf2dGaussianHessian(np.array([pt_temp]), params)[0]
    #         eig_vals, eig_vecs = np.linalg.eig(Hessian)
    #         if all(eig_vals > 0):
    #             criticalPts_types.append(1)
    #         elif all(eig_vals < 0):
    #             criticalPts_types.append(2)
    #         else:
    #             criticalPts_types.append(3)

    return criticalPts, criticalPts_types, num_cps, mesh_nodes, mesh_nodes_labels, num_mesh_nodes


def MorseRegions(params, xx, yy, x_dim, y_dim):
    grid_grad = mixtureOf2dGaussianFieldGradient(params, xx, yy, x_dim, y_dim)
    norm_grid_grad = np.linalg.norm(grid_grad, axis=2)
    grid_Hessian = mixtureOf2dGaussianFieldHessian(params, xx, yy, x_dim, y_dim)
    eigvals = np.linalg.eigvals(grid_Hessian)
    Morse_region_1 = norm_grid_grad > degenerate_threshold_gradient
    Morse_region_2 = np.logical_or(np.abs(eigvals[:, :, 0]) > degenerate_threshold_eigvals,
                                   np.abs(eigvals[:, :, 1]) > degenerate_threshold_eigvals)
    Morse_region = np.logical_or(Morse_region_1, Morse_region_2)
    Morse_region = Morse_region.astype(int)
    # ax = plt.figure().add_subplot()
    # plt.pcolormesh(x_space, y_space, grid_func, cmap='Greys')
    # plt.pcolormesh(x_space, y_space, Morse_region, cmap='Greens', alpha=0.1)
    # plt.show()
    return Morse_region


def ifMorsePoints(pts: np.array, params):
    grads = mixtureOf2dGaussianGradient(pts, params)
    norm_grads = np.linalg.norm(grads, axis=1)
    Hessians = mixtureOf2dGaussianHessian(pts, params)
    eigvals = np.linalg.eigvals(Hessians)
    Morse_pts_1 = norm_grads > degenerate_threshold_gradient
    Morse_pts_2 = np.logical_or(np.abs(eigvals[:, 0]) > degenerate_threshold_eigvals,
                                np.abs(eigvals[:, 1]) > degenerate_threshold_eigvals)
    Morse_pts = np.logical_or(Morse_pts_1, Morse_pts_2)
    return Morse_pts


def adpativeSeeds(x_dim, y_dim, x_range, y_range, x_space, y_space, grid_func, params, time_interval, round_num_seeds,
                  avg_num_samples, x_bsn, y_bsn, Morse_region):
    '''

    Args:
        x_bsn: the number of seeds on field boundary along x direction
        y_bsn: the number of seeds on field boundary along y direction

    Returns:

    '''
    # generating initial seeds
    rng = np.random.default_rng()
    r = 0.2
    engine = PoissonDisk(d=2, radius=r, seed=rng)
    seeds_tmp = engine.random(round_num_seeds)
    seeds_tmp[:, 0] = (seeds_tmp[:, 0] * (x_range[1] - x_range[0]) + x_range[0]) * 0.8
    seeds_tmp[:, 1] = (seeds_tmp[:, 1] * (y_range[1] - y_range[0]) + y_range[0]) * 0.8
    num_seeds_tmp = seeds_tmp.shape[0]
    num_seeds_bound = 2 * (x_bsn + y_bsn - 2)
    num_seeds = num_seeds_tmp + num_seeds_bound
    seeds = np.zeros((num_seeds, 2))
    seeds[:num_seeds_tmp] = seeds_tmp
    x_bs = np.linspace(x_range[0], x_range[1], x_bsn)
    y_bs = np.linspace(y_range[0], y_range[1], y_bsn)
    seeds[num_seeds_tmp:num_seeds_tmp + x_bsn, 0] = x_bs
    seeds[num_seeds_tmp:num_seeds_tmp + x_bsn, 1] = np.ones(x_bsn) * y_range[0]
    seeds[num_seeds_tmp + x_bsn:num_seeds_tmp + 2 * x_bsn, 0] = x_bs
    seeds[num_seeds_tmp + x_bsn:num_seeds_tmp + 2 * x_bsn, 1] = np.ones(x_bsn) * y_range[1]
    seeds[num_seeds_tmp + 2 * x_bsn:num_seeds_tmp + 2 * x_bsn + y_bsn - 2, 0] = np.ones(y_bsn - 2) * x_range[0]
    seeds[num_seeds_tmp + 2 * x_bsn:num_seeds_tmp + 2 * x_bsn + y_bsn - 2, 1] = y_bs[1:-1]
    seeds[num_seeds_tmp + 2 * x_bsn + y_bsn - 2:, 0] = np.ones(y_bsn - 2) * x_range[1]
    seeds[num_seeds_tmp + 2 * x_bsn + y_bsn - 2:, 1] = y_bs[1:-1]

    inMorse = ifMorsePoints(seeds, params)
    Morse_seeds_idx = np.where(inMorse)[0]
    num_Morse_seeds = Morse_seeds_idx.shape[0]
    Morse_seeds = seeds[Morse_seeds_idx]
    fn = "streamlines/" + str(round_num_seeds) + "_" + str(avg_num_samples) + "_" + str(time_interval) + ".npy"
    if os.path.isfile(fn):
        with open(fn, "rb") as f:
            streamlines = np.load(f)
    else:
        streamlines = np.zeros((num_Morse_seeds, avg_num_samples + 1, 2))
        for i, s in enumerate(Morse_seeds):
            streamline = streamline_trace(s, params, time_interval, avg_num_samples)
            streamlines[i] = streamline
        with open(fn, "wb") as f:
            np.save(f, streamlines)
    criticalPts_tmp, criticalPts_types, samples_virtual, mesh_nodes_labels \
        = criticalPointsFromStreamlines(streamlines, params, time_interval, avg_num_samples)

    fig_id = 0
    num_cps = criticalPts_tmp.shape[0]
    criticalPts = np.zeros((100, 2))
    criticalPts[:num_cps] = criticalPts_tmp
    num_samples = samples_virtual.shape[0]
    num_mesh_nodes = num_samples
    mesh_nodes = np.zeros((1000000000, 2))
    mesh_nodes[:num_mesh_nodes] = samples_virtual
    delaunay = Delaunay(mesh_nodes[:num_mesh_nodes])
    tris = delaunay.simplices
    trisPlot2D(tris, mesh_nodes[:num_mesh_nodes], streamlines, criticalPts[:num_cps], criticalPts_types, x_space,
               y_space, grid_func, Morse_region, fig_id)
    Morse_seeds_idx_set = set(Morse_seeds_idx)
    hidden_tris = []

    while num_seeds < 100:
        fig_id += 1
        tri_weights = np.ones(tris.shape[0])
        for i, tri in enumerate(tris):
            num_safe_nodes = len(Morse_seeds_idx_set.intersection(set(list(tri))))
            if num_safe_nodes == 0:
                tri_weights[i] = 0
            else:
                tri_weights[i] *= 0.1 ** (3 - num_safe_nodes)
            mat = np.zeros((2, 2))
            mat[0] = mesh_nodes[tri[1]] - mesh_nodes[tri[0]]
            mat[1] = mesh_nodes[tri[2]] - mesh_nodes[tri[0]]
            tri_area = np.abs(np.linalg.det(mat)) / 2
            tri_weights[i] *= tri_area
        # tri = np.argmax(tri_weights)
        idx = len(hidden_tris) + 1
        tri = np.argsort(tri_weights, axis=0)[-idx]

        center = circle(mesh_nodes[tris[tri]])
        ifMorse = ifMorsePoints(center.reshape(-1, 2), params)[0]
        if center[0] < -20 or center[0] > 20 or center[1] < -20 or center[1] > 20:
            hidden_tris.append(tri)
        else:
            # mesh_nodes[num_mesh_nodes] = center
            # num_mesh_nodes += 1
            if ifMorse:
                Morse_seeds_idx_set.add(num_seeds)
                streamline = streamline_trace(center, params, time_interval, avg_num_samples)
                criticalPts, criticalPts_types, num_cps, mesh_nodes, mesh_nodes_labels, num_mesh_nodes = \
                    updateCriticalPointsFromOneStreamline(streamline, criticalPts, criticalPts_types, num_cps,
                                                          mesh_nodes, mesh_nodes_labels, num_mesh_nodes, params,
                                                          time_interval, avg_num_samples)
            else:
                mesh_nodes_labels.append(None)
            num_seeds += 1
            delaunay = Delaunay(mesh_nodes[:num_mesh_nodes])
            tris = delaunay.simplices
            tris_label = []
            for tri in tris:
                if mesh_nodes_labels[tri[0]] == mesh_nodes_labels[tri[1]] and mesh_nodes_labels[tri[0]] == \
                        mesh_nodes_labels[tri[2]]:
                    tris_label.append(0)
                elif mesh_nodes_labels[tri[0]] is None or mesh_nodes_labels[tri[1]] is None or mesh_nodes_labels[
                    tri[2]] is None:
                    tris_label.append(0)
                else:
                    tris_label.append(1)
            # trisPlot2D(tris, mesh_nodes[:num_mesh_nodes], streamlines, criticalPts[:num_cps], criticalPts_types,
            #            x_space, y_space, grid_func, Morse_region, fig_id)
            trisPlotColor(tris, tris_label, mesh_nodes[:num_mesh_nodes], streamlines, criticalPts[:num_cps],
                          criticalPts_types, x_space, y_space, grid_func, Morse_region, fig_id)


if __name__ == "__main__":
    x_space = np.linspace(x_range[0], x_range[1], x_dim)
    y_space = np.linspace(y_range[0], y_range[1], y_dim)
    xx, yy = np.meshgrid(x_space, y_space)
    params = [[-8, -8, fwhm2sigma(20), fwhm2sigma(20), 0, 40 * 25],
              [0, 10, fwhm2sigma(10), fwhm2sigma(10), 0, 10 * 25],
              [5, -3, fwhm2sigma(8), fwhm2sigma(8), 0, 3 * 25]]
    grid_func = mixtureOf2dGaussianField(params, xx, yy, x_dim, y_dim)
    Morse_region = MorseRegions(params, xx, yy, x_dim, y_dim)
    adpativeSeeds(x_dim, y_dim, x_range, y_range, x_space, y_space, grid_func, params, time_interval, round_num_seeds,
                  avg_num_samples, 4, 4, Morse_region)
