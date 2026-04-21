#!/usr/bin/env python3
"""
Interactive KITTI LiDAR + 3D label viewer (Open3D) with tighter visual boxes
and mild ground suppression for in-box clustering/highlighting.

What is updated:
- Mild, robust ground suppression inside each object box
- Optional box tightening from enclosed points without being too aggressive
- Tighter highlighting mask so red points do not include floor stripes
- Keeps original workflow and keyboard controls

Keys:
  D / Right Arrow  : next frame
  A / Left Arrow   : previous frame
  R                : reload current frame
  Y                : toggle yaw usage (use/ignore ry)
  S                : save current frame screenshot
  B                : toggle autosave (on navigation)
  K                : lock/unlock current camera view
  Q or ESC         : quit
"""

import argparse
from pathlib import Path
import time
import copy
import numpy as np
import open3d as o3d


# -------------------------
# Parsing KITTI labels
# -------------------------
def parse_kitti_label_file(label_path: Path):
    """
    Returns list of dicts: {type, hwl=(h,w,l), xyz=(x,y,z), ry}
    KITTI label_2 format:
      type trunc occl alpha bbox_l bbox_t bbox_r bbox_b h w l x y z ry
    """
    objs = []
    if not label_path.exists():
        return objs

    txt = label_path.read_text().strip()
    if not txt:
        return objs

    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 15:
            continue
        cls = parts[0]

        try:
            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            x = float(parts[11])
            y = float(parts[12])
            z = float(parts[13])
            ry = float(parts[14])
        except Exception:
            continue

        objs.append({
            "type": cls,
            "hwl": (h, w, l),
            "xyz": (x, y, z),
            "ry": ry,
        })
    return objs


# -------------------------
# Geometry helpers
# -------------------------
def rotz(ry: float):
    c = float(np.cos(ry))
    s = float(np.sin(ry))
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def make_box_lineset(center, hwl, ry, color=(1.0, 0.55, 0.0), ignore_ry=False):
    h, w, l = hwl
    extent = np.array([l, w, h], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) if ignore_ry else rotz(ry)
    obb = o3d.geometry.OrientedBoundingBox(
        center=np.array(center, dtype=np.float32),
        R=R.astype(np.float64),
        extent=extent.astype(np.float64),
    )
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lineset.paint_uniform_color(color)
    return lineset


def world_to_local(pts_xyz, center, ry, ignore_ry=False):
    if pts_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    q = pts_xyz.astype(np.float32) - c[None, :]
    if (not ignore_ry) and abs(float(ry)) > 1e-8:
        R = rotz(-ry)
        q = q @ R.T
    return q


def local_to_world(pts_local, center, ry, ignore_ry=False):
    if pts_local.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    q = pts_local.astype(np.float32)
    if (not ignore_ry) and abs(float(ry)) > 1e-8:
        R = rotz(ry)
        q = q @ R.T
    return q + c[None, :]


def points_in_oriented_box(pts_xyz, center, hwl, ry, ignore_ry=False, margin_xyz=(0.0, 0.0, 0.0)):
    if pts_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    h, w, l = hwl
    q = world_to_local(pts_xyz, center, ry, ignore_ry=ignore_ry)
    mx, my, mz = [float(v) for v in margin_xyz]
    half = np.array([
        max(l / 2.0 + mx, 1e-4),
        max(w / 2.0 + my, 1e-4),
        max(h / 2.0 + mz, 1e-4),
    ], dtype=np.float32)
    inside = np.all(np.abs(q) <= (half[None, :] + 1e-6), axis=1)
    return inside


def robust_percentile_bounds(x, lo=5.0, hi=95.0):
    if x.size == 0:
        return None, None
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def clamp_shrink(old_extent, new_extent, max_shrink_ratio):
    """
    Only allow shrinking. Limit how much shrinking is applied.
    max_shrink_ratio=0.18 means new extent cannot be smaller than 82% of old extent.
    """
    min_extent = old_extent * (1.0 - float(max_shrink_ratio))
    new_extent = min(old_extent, new_extent)
    new_extent = max(min_extent, new_extent)
    return new_extent


def fit_tighter_box_from_points(
    pts_xyz,
    center,
    hwl,
    ry,
    ignore_ry=False,
    fit_min_points=20,
    fit_percentile_lo=10.0,
    fit_percentile_hi=90.0,
    fit_max_shrink_xy=0.30,
    fit_max_shrink_z=0.18,
    fit_center_blend=0.92,
    ground_percentile=10.0,
    ground_margin=0.06,
    leg_recover_margin=0.05,
    highlight_shrink_xy=0.06,
):
    """
    Stronger controlled tightening with softer lower-body suppression.

    Update:
    - keeps the box tight in XY
    - removes only the lowest floor strip
    - recovers a small band above that floor estimate so legs are not cut
    """
    raw_mask = points_in_oriented_box(pts_xyz, center, hwl, ry, ignore_ry=ignore_ry)
    if not raw_mask.any():
        return {
            "display_center": tuple(center),
            "display_hwl": tuple(hwl),
            "display_mask": raw_mask,
            "nonground_mask": raw_mask,
            "raw_mask": raw_mask,
        }

    pts_raw = pts_xyz[raw_mask]
    pts_local = world_to_local(pts_raw, center, ry, ignore_ry=ignore_ry)

    # estimate the lowest local floor band inside the box
    z_floor = float(np.percentile(pts_local[:, 2], ground_percentile))

    # soft suppression only: keep points slightly above floor so ankles/legs survive
    keep_threshold = z_floor + float(ground_margin)
    nonground_local_mask = pts_local[:, 2] > keep_threshold

    if int(np.count_nonzero(nonground_local_mask)) >= max(10, fit_min_points // 2):
        pts_fit = pts_local[nonground_local_mask]
    else:
        pts_fit = pts_local
        nonground_local_mask = np.ones((pts_local.shape[0],), dtype=bool)

    if pts_fit.shape[0] < fit_min_points:
        tight_mask = points_in_oriented_box(
            pts_xyz,
            center,
            hwl,
            ry,
            ignore_ry=ignore_ry,
            margin_xyz=(-highlight_shrink_xy, -highlight_shrink_xy, 0.0),
        )
        display_mask = np.zeros_like(raw_mask)
        display_mask[np.where(raw_mask)[0][nonground_local_mask]] = True
        display_mask &= tight_mask
        return {
            "display_center": tuple(center),
            "display_hwl": tuple(hwl),
            "display_mask": display_mask,
            "nonground_mask": display_mask,
            "raw_mask": raw_mask,
        }

    x0, x1 = robust_percentile_bounds(pts_fit[:, 0], fit_percentile_lo, fit_percentile_hi)
    y0, y1 = robust_percentile_bounds(pts_fit[:, 1], fit_percentile_lo, fit_percentile_hi)
    z0, z1 = robust_percentile_bounds(pts_fit[:, 2], fit_percentile_lo, fit_percentile_hi)

    old_h, old_w, old_l = [float(v) for v in hwl]

    pad_x = 0.04
    pad_y = 0.04
    pad_z_top = 0.03
    pad_z_bottom = 0.06

    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y
    z0 -= pad_z_bottom
    z1 += pad_z_top

    new_l = max(0.10, x1 - x0)
    new_w = max(0.10, y1 - y0)
    new_h = max(0.18, z1 - z0)

    new_l = clamp_shrink(old_l, new_l, fit_max_shrink_xy)
    new_w = clamp_shrink(old_w, new_w, fit_max_shrink_xy)
    new_h = clamp_shrink(old_h, new_h, fit_max_shrink_z)

    local_center_old = np.zeros((1, 3), dtype=np.float32)
    local_center_new = np.array([[(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5]], dtype=np.float32)
    local_center_blend = (1.0 - float(fit_center_blend)) * local_center_old + float(fit_center_blend) * local_center_new
    display_center = local_to_world(local_center_blend, center, ry, ignore_ry=ignore_ry)[0]
    display_hwl = (new_h, new_w, new_l)

    display_mask = points_in_oriented_box(
        pts_xyz,
        display_center,
        display_hwl,
        ry,
        ignore_ry=ignore_ry,
        margin_xyz=(0.0, 0.0, 0.0),
    )

    display_mask_tight = points_in_oriented_box(
        pts_xyz,
        display_center,
        display_hwl,
        ry,
        ignore_ry=ignore_ry,
        margin_xyz=(-highlight_shrink_xy, -highlight_shrink_xy, 0.0),
    )

    pts_disp_local = world_to_local(pts_xyz[display_mask], display_center, ry, ignore_ry=ignore_ry)
    if pts_disp_local.shape[0] > 0:
        z_floor_disp = float(np.percentile(pts_disp_local[:, 2], ground_percentile))
        base_keep = pts_disp_local[:, 2] > (z_floor_disp + float(ground_margin))

        # recover a thin lower band if it is vertically connected to the body cluster
        recover_band = (
            (pts_disp_local[:, 2] > (z_floor_disp + max(0.0, float(ground_margin) - float(leg_recover_margin))))
            & (pts_disp_local[:, 2] <= (z_floor_disp + float(ground_margin)))
        )
        near_center_y = np.abs(pts_disp_local[:, 1]) <= (display_hwl[1] * 0.28)
        leg_recover = recover_band & near_center_y

        keep_disp_local = base_keep | leg_recover
        nonground_mask = np.zeros_like(display_mask)
        nonground_mask[np.where(display_mask)[0][keep_disp_local]] = True
    else:
        nonground_mask = display_mask.copy()

    display_mask = display_mask_tight & nonground_mask

    return {
        "display_center": tuple(float(v) for v in display_center),
        "display_hwl": tuple(float(v) for v in display_hwl),
        "display_mask": display_mask,
        "nonground_mask": nonground_mask,
        "raw_mask": raw_mask,
    }


# -------------------------
# Point cloud loading / filtering
# -------------------------
def load_velodyne_bin(bin_path: Path):
    arr = np.fromfile(str(bin_path), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    arr = arr.reshape(-1, 4)
    return arr


def filter_range_360(pts_xyz, range_m=50.0, use_xy=True, z_min=None, z_max=None):
    if pts_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    if use_xy:
        r = np.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2)
    else:
        r = np.linalg.norm(pts_xyz, axis=1)

    m = (r <= float(range_m))
    if z_min is not None:
        m &= (pts_xyz[:, 2] >= float(z_min))
    if z_max is not None:
        m &= (pts_xyz[:, 2] <= float(z_max))
    return m


def subsample_random(pts_xyz, pts_i, max_points=200000, seed=0):
    if pts_xyz.shape[0] <= max_points:
        return pts_xyz, pts_i
    rng = np.random.RandomState(seed)
    idx = rng.choice(pts_xyz.shape[0], size=max_points, replace=False)
    return pts_xyz[idx], (pts_i[idx] if pts_i is not None else None)


# -------------------------
# Viewer class
# -------------------------
class Kitti3DViewer:
    def __init__(self, args):
        self.args = args
        self.velodyne_dir = Path(args.velodyne_dir)
        self.label_dir = Path(args.label_dir)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        bin_ids = sorted([p.stem for p in self.velodyne_dir.glob("*.bin")])
        if not bin_ids:
            raise RuntimeError(f"No .bin files in {self.velodyne_dir}")

        if args.only_with_labels:
            lab_ids = set([p.stem for p in self.label_dir.glob("*.txt")])
            self.ids = [i for i in bin_ids if i in lab_ids]
        else:
            self.ids = bin_ids

        if args.start_id is not None:
            if args.start_id in self.ids:
                self.idx = self.ids.index(args.start_id)
            else:
                raise RuntimeError(f"--start-id {args.start_id} not found in ids.")
        else:
            self.idx = 0

        self.ignore_ry = args.ignore_ry
        self.autosave = args.autosave
        self.lock_view = False
        self._locked_cam = None

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self.box_sets = []

        self._init_vis()
        self._load_current(add_geometry_first=True)

        if args.lock_view:
            self._lock_current_view()

    def _init_vis(self):
        self.vis.create_window(
            window_name="KITTI 3D Label Viewer | frame info in console and saved filename",
            width=self.args.win_w,
            height=self.args.win_h,
        )

        self.vis.register_key_callback(ord("D"), self._cb_next)
        self.vis.register_key_callback(ord("A"), self._cb_prev)
        self.vis.register_key_callback(ord("R"), self._cb_reload)
        self.vis.register_key_callback(ord("Y"), self._cb_toggle_yaw)
        self.vis.register_key_callback(ord("S"), self._cb_save)
        self.vis.register_key_callback(ord("B"), self._cb_toggle_autosave)
        self.vis.register_key_callback(ord("K"), self._cb_toggle_lock_view)
        self.vis.register_key_callback(ord("Q"), self._cb_quit)
        self.vis.register_key_callback(256, self._cb_quit)
        self.vis.register_key_callback(262, self._cb_next)
        self.vis.register_key_callback(263, self._cb_prev)

        opt = self.vis.get_render_option()
        opt.background_color = np.array(self.args.bg, dtype=np.float32)
        opt.point_size = float(self.args.point_size)
        opt.line_width = float(self.args.box_line_width)

    def _current_paths(self):
        fid = self.ids[self.idx]
        bin_path = self.velodyne_dir / f"{fid}.bin"
        label_path = self.label_dir / f"{fid}.txt"
        return fid, bin_path, label_path

    def _clear_boxes(self):
        for ls in self.box_sets:
            self.vis.remove_geometry(ls, reset_bounding_box=False)
        self.box_sets = []

    def _maybe_apply_locked_view(self):
        if self.lock_view and self._locked_cam is not None:
            vc = self.vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(self._locked_cam, allow_arbitrary=True)

    def _lock_current_view(self):
        vc = self.vis.get_view_control()
        self._locked_cam = vc.convert_to_pinhole_camera_parameters()
        self.lock_view = True
        print("[VIEW] Camera LOCKED (K to toggle).")

    def _unlock_view(self):
        self.lock_view = False
        print("[VIEW] Camera UNLOCKED (K to toggle).")

    def _save_screenshot(self, fid: str):
        self._maybe_apply_locked_view()
        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(0.02)
        out_path = self.save_dir / f"frame_{self.idx:06d}_{fid}.png"
        ok = self.vis.capture_screen_image(str(out_path), do_render=True)
        if ok:
            print(f"[SAVE] {out_path}")
        else:
            print(f"[WARN] capture_screen_image returned False for {out_path}")

    def _load_current(self, add_geometry_first=False, do_autosave=False):
        fid, bin_path, label_path = self._current_paths()

        pts4 = load_velodyne_bin(bin_path)
        xyz = pts4[:, :3].astype(np.float32)
        intensity = pts4[:, 3].astype(np.float32) if pts4.shape[1] == 4 else None

        m = filter_range_360(
            xyz,
            range_m=self.args.range_m,
            use_xy=True,
            z_min=self.args.z_min,
            z_max=self.args.z_max,
        )
        xyz = xyz[m]
        intensity = intensity[m] if intensity is not None else None

        xyz, intensity = subsample_random(xyz, intensity, max_points=self.args.max_points, seed=self.idx)

        objs = parse_kitti_label_file(label_path)
        if self.args.class_filter:
            keep = set(self.args.class_filter)
            objs = [o for o in objs if o["type"] in keep]

        colors = np.tile(
            np.array(self.args.bg_cloud, dtype=np.float32)[None, :],
            (xyz.shape[0], 1),
        )

        palette = np.array([
            [1.00, 0.20, 0.20],
            [0.20, 1.00, 0.20],
            [0.20, 0.60, 1.00],
            [1.00, 0.60, 0.20],
            [0.90, 0.20, 0.90],
            [0.20, 0.90, 0.90],
        ], dtype=np.float32)

        self._clear_boxes()
        fitted_count = 0

        for k, o in enumerate(objs):
            disp_center = o["xyz"]
            disp_hwl = o["hwl"]
            disp_mask = points_in_oriented_box(
                xyz,
                center=disp_center,
                hwl=disp_hwl,
                ry=float(o["ry"]),
                ignore_ry=self.ignore_ry,
            )

            if self.args.tighten_boxes:
                fit = fit_tighter_box_from_points(
                    xyz,
                    center=o["xyz"],
                    hwl=o["hwl"],
                    ry=float(o["ry"]),
                    ignore_ry=self.ignore_ry,
                    fit_min_points=self.args.fit_min_points,
                    fit_percentile_lo=self.args.fit_percentile_lo,
                    fit_percentile_hi=self.args.fit_percentile_hi,
                    fit_max_shrink_xy=self.args.fit_max_shrink_xy,
                    fit_max_shrink_z=self.args.fit_max_shrink_z,
                    fit_center_blend=self.args.fit_center_blend,
                    ground_percentile=self.args.ground_percentile,
                    ground_margin=self.args.ground_margin,
                    leg_recover_margin=self.args.leg_recover_margin,
                    highlight_shrink_xy=self.args.highlight_shrink_xy,
                )
                disp_center = fit["display_center"]
                disp_hwl = fit["display_hwl"]
                disp_mask = fit["display_mask"]
                fitted_count += int(np.any(fit["raw_mask"]))

            if disp_mask.any():
                if self.args.per_box_color:
                    colors[disp_mask] = palette[k % len(palette)]
                else:
                    colors[disp_mask] = np.array(self.args.in_color, dtype=np.float32)

            ls = make_box_lineset(
                center=disp_center,
                hwl=disp_hwl,
                ry=float(o["ry"]),
                color=self.args.box_color,
                ignore_ry=self.ignore_ry,
            )
            self.box_sets.append(ls)

        self.pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        if add_geometry_first:
            self.vis.add_geometry(self.pcd, reset_bounding_box=True)
            for ls in self.box_sets:
                self.vis.add_geometry(ls, reset_bounding_box=False)
        else:
            self.vis.update_geometry(self.pcd)
            for ls in self.box_sets:
                self.vis.add_geometry(ls, reset_bounding_box=False)

        self._maybe_apply_locked_view()

        n_boxes = len(objs)
        n_pts = xyz.shape[0]
        yaw_state = "IGNORED" if self.ignore_ry else "USED"
        print(
            f"[VIEW] frame={self.idx + 1}/{len(self.ids)} id={fid} points={n_pts} boxes={n_boxes} yaw={yaw_state} "
            f"range={self.args.range_m}m autosave={'ON' if self.autosave else 'OFF'} "
            f"lock_view={'ON' if self.lock_view else 'OFF'} tighten={'ON' if self.args.tighten_boxes else 'OFF'} "
            f"ground_margin={self.args.ground_margin:.2f} leg_recover={self.args.leg_recover_margin:.2f} fitted={fitted_count}"
        )

        self.vis.poll_events()
        self.vis.update_renderer()

        if do_autosave and self.autosave:
            self._save_screenshot(fid)

    def _cb_next(self, vis):
        self.idx = min(self.idx + 1, len(self.ids) - 1)
        self._load_current(do_autosave=True)
        return False

    def _cb_prev(self, vis):
        self.idx = max(self.idx - 1, 0)
        self._load_current(do_autosave=True)
        return False

    def _cb_reload(self, vis):
        self._load_current(do_autosave=True)
        return False

    def _cb_toggle_yaw(self, vis):
        self.ignore_ry = not self.ignore_ry
        self._load_current(do_autosave=True)
        return False

    def _cb_save(self, vis):
        fid, _, _ = self._current_paths()
        self._save_screenshot(fid)
        return False

    def _cb_toggle_autosave(self, vis):
        self.autosave = not self.autosave
        print(f"[SAVE] autosave={'ON' if self.autosave else 'OFF'} (B to toggle)")
        return False

    def _cb_toggle_lock_view(self, vis):
        if self.lock_view:
            self._unlock_view()
        else:
            self._lock_current_view()
        self._load_current(do_autosave=False)
        return False

    def _cb_quit(self, vis):
        vis.close()
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--velodyne-dir", required=True, help="KITTI training/velodyne directory")
    ap.add_argument("--label-dir", required=True, help="KITTI training/label_2 directory")
    ap.add_argument("--start-id", default=None, help="e.g., 000123")
    ap.add_argument("--only-with-labels", action="store_true", help="Only show frames that have label_2 txt")

    ap.add_argument("--range-m", type=float, default=50.0, help="360° range crop radius in meters (XY)")
    ap.add_argument("--z-min", type=float, default=None, help="Optional z min clamp")
    ap.add_argument("--z-max", type=float, default=None, help="Optional z max clamp")
    ap.add_argument("--max-points", type=int, default=180000, help="Random subsample after range crop")

    ap.add_argument("--ignore-ry", action="store_true", help="Ignore yaw ry (axis-aligned boxes)")
    ap.add_argument("--per-box-color", action="store_true", help="Different inside-point color per box")
    ap.add_argument("--in-color", type=float, nargs=3, default=[1.0, 0.2, 0.2], help="Inside-box color if not per-box")
    ap.add_argument("--bg-cloud", type=float, nargs=3, default=[0.70, 0.78, 0.92], help="Background cloud color")
    ap.add_argument("--box-color", type=float, nargs=3, default=[1.0, 0.55, 0.0], help="3D box line color")
    ap.add_argument("--class-filter", nargs="*", default=None, help="e.g., Pedestrian Person Car")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--box-line-width", type=float, default=2.5)
    ap.add_argument("--bg", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Background color")
    ap.add_argument("--win-w", type=int, default=1400)
    ap.add_argument("--win-h", type=int, default=800)

    # New tightening / ground suppression controls
    ap.add_argument("--tighten-boxes", action="store_true", default=True,
                    help="Mildly tighten visual boxes from enclosed points")
    ap.add_argument("--fit-min-points", type=int, default=20,
                    help="Minimum enclosed points required before fitting tighter box")
    ap.add_argument("--fit-percentile-lo", type=float, default=10.0,
                    help="Lower percentile used for robust box fitting")
    ap.add_argument("--fit-percentile-hi", type=float, default=90.0,
                    help="Upper percentile used for robust box fitting")
    ap.add_argument("--fit-max-shrink-xy", type=float, default=0.30,
                    help="Maximum allowed XY shrink ratio for fitted box")
    ap.add_argument("--fit-max-shrink-z", type=float, default=0.18,
                    help="Maximum allowed height shrink ratio for fitted box")
    ap.add_argument("--fit-center-blend", type=float, default=0.92,
                    help="Blend fitted center with original center, higher means tighter recentering")
    ap.add_argument("--ground-percentile", type=float, default=10.0,
                    help="Local lower percentile used to estimate floor band inside a box")
    ap.add_argument("--ground-margin", type=float, default=0.06,
                    help="Height margin above local floor to reject ground points inside a box")
    ap.add_argument("--leg-recover-margin", type=float, default=0.05,
                    help="Recover a thin lower band above the removed floor so legs are preserved")
    ap.add_argument("--highlight-shrink-xy", type=float, default=0.06,
                    help="Extra XY shrink used only for highlighted in-box points")

    ap.add_argument("--save-dir", default="saved_frames", help="Where screenshots will be written")
    ap.add_argument("--autosave", action="store_true", help="Save frame on next/prev/reload")
    ap.add_argument("--lock-view", action="store_true", help="Start with camera view locked")

    args = ap.parse_args()

    v = Kitti3DViewer(args)
    v.run()


if __name__ == "__main__":
    main()
