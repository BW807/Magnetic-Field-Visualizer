"""
CONTROLS
    Left click   add a vertex (snapped)        n  finish wire, start a new one
    Right click  undo last vertex              u  undo last vertex
    Clear all    remove everything             c  clear all
Current flows from each wire's first vertex to its last.

CODE LAYOUT
-----------
    splitwires()       breaks wires into small segments (midpoints + dl vectors)
    field_per_amp()    segments  -> B field at grid points for 1 A  (slow computationally)
    WireSim class      GUI: drawing, widgets

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import Button, RadioButtons, Slider

# ---------------------------------------------------------------- constants
MU0 = 4e-7 * np.pi        # vacuum permeability, T*m/A
WIRE_RADIUS = 0.5e-3      # m; softens 1/r^2 so points on a wire stay finite
N_EVAL = 41               # field grid is N_EVAL x N_EVAL points
MAX_SEGS = 4000           # cap on total segments (bounds the rebuild time)
N_DOTS = 6               # flow dots per wire
DOT_SPEED = 0.01          # fraction of wire length per frame at full current
DT = 0.04                 # simulated seconds per frame (matches interval=40 ms)
SNAPS = {"1 m": 1.0, "1 cm": 0.01, "1 mm": 0.001}

def splitwires(wire, seg_max):
    """Split a wire into straight pieces no longer than `seg_max`"""
    # Empty lists that will collect each edge's midpoints and dl vectors
    mids, dls = [], []

    # Walk through the wire one edge at a time: edge p -> q, where p is a vertex and q is the next one
    for p, q in zip(wire[:-1], wire[1:]):

        # Displacement vector along this edge, pointing in the direction of current flow
        d = q - p

        # Length of the edge in metres.
        length = np.hypot(*d)

        # Skip zero-length edges (two identical vertices), since they carry no current
        if length == 0:
            continue

        # Number of pieces needed so that none is longer than seg_max
        k = int(np.ceil(length / seg_max))

        # Fractional positions of each piece's midpoint along the edge, from 0 to 1
        # For k = 2 this gives [0.25, 0.75], the centre of each half
        t = (np.arange(k) + 0.5) / k

        # Convert those fractions into actual (x, y) midpoints: start at p, move t along d
        mids.append(p + t[:, None] * d)

        # Each piece has the same vector, d / k, so repeat that row k times
        dls.append(np.tile(d / k, (k, 1)))

    # If every edge was skipped (or the wire has fewer than two vertices), return empty arrays of the right shape
    if not mids:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Stack the per-edge arrays into single (M, 2) arrays covering all pieces of all edges
    return np.vstack(mids), np.vstack(dls)
    #Returns : mids (M, 2) piece midpoints, dls (M, 2) piece vectors dl (dl points along the current direction).

def field_per_amp(px, py, h, mids, dls, chunk=256):
    """This is the part of the code doing the biot-savart law work!"""
    
    # Create three zero arrays, one per field component, each with one entry per grid point
    Bx, By, Bz = (np.zeros_like(px) for _ in range(3))

    # Process the segments in blocks of chunks (256, defined in the function) to keep temporary arrays small
    for i in range(0, len(mids), chunk):
        # Take this block's segment midpoints (m) and current-direction vectors (d)
        m, d = mids[i:i + chunk], dls[i:i + chunk]

        # Horizontal distance from every grid point to every segment midpoint, shape (P, chunk)
        rx = px[:, None] - m[None, :, 0]
        ry = py[:, None] - m[None, :, 1]

        # 1 / (R² + a²)^(3/2), where R² = rx² + ry² + h² is the squared 3D distance
        # and a² = WIRE_RADIUS² as a failsafe so the radius is never 0
        inv_r3 = (rx ** 2 + ry ** 2 + h ** 2 + WIRE_RADIUS ** 2) ** -1.5

        # Current-direction components of each segment, shape (1, chunk) so they broadcast over grid points
        dx, dy = d[None, :, 0], d[None, :, 1]

        # x-component of dl × R, which is dy·h, times the kernel. Sum over segments, add to the running total
        Bx += (dy * h * inv_r3).sum(axis=1)
        # Notice this is just like the x-component, but the cross product dl × R results in -dx * h for y
        By += (-dx * h * inv_r3).sum(axis=1)
        Bz += ((dx * ry - dy * rx) * inv_r3).sum(axis=1)
    k = MU0 / (4 * np.pi)

    # Return three 1D arrays, each of length P, one value per grid point
    # These values are complied into a matrix later on (recompute function)
    return k * Bx, k * By, k * Bz

class WireSim:
    """This is the class responsible for the visuals, not the physics."""

    def __init__(self):
        self.wires = [[]]          # list of vertex lists; the last one is active
        self.cursor = None         # snapped mouse position (x, y) or None
        self.dirty = True          # True -> field must be recomputed
        self.B = np.zeros((3, N_EVAL, N_EVAL))   # field for 1 A (Bx, By, Bz)
        self.paths = []            # per wire: (vertices, cumulative length)
        self.phase = 0.0           # flow-dot phase, fraction of a wire
        self.log_scale = False     # which colour norm the image currently has

        self._build_figure()
        self._build_widgets()
        self._connect_events()
        self.draw_wires()

    # ------------------------------------------------------------- set-up
    def _build_figure(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_axes([0.2, 0.3, 0.6, 0.66])
        half = 5.0
        self.im = self.ax.imshow(np.ones((N_EVAL, N_EVAL)), origin="lower",
                                 extent=[-half, half, -half, half],
                                 cmap="viridis", norm=Normalize(0, 1))
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="|B| (µT)")
        zeros = np.zeros(N_EVAL * N_EVAL)
        self.quiver = self.ax.quiver(zeros, zeros, zeros, zeros, angles="xy",
                                     scale_units="xy", scale=1, pivot="mid",
                                     color="white", width=0.003)
        # All wires in ONE line: NaN entries break it between wires.
        self.wire_line, = self.ax.plot([], [], "ro-", lw=2, ms=4, zorder=3)
        self.preview, = self.ax.plot([], [], "r--", lw=1, zorder=3)
        self.cursor_dot, = self.ax.plot([], [], "k+", ms=12, zorder=5)
        self.flow, = self.ax.plot([], [], "o", ms=5, color="yellow", zorder=6)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.title = self.ax.set_title("")

    def _build_widgets(self):
        f = self.fig
        self.r_snap = RadioButtons(f.add_axes([0.01, 0.72, 0.1, 0.13]),
                                   list(SNAPS))
        self.r_mode = RadioButtons(f.add_axes([0.01, 0.55, 0.1, 0.12]),
                                   ("DC", "AC"), active=1)
        self.r_scale = RadioButtons(f.add_axes([0.01, 0.40, 0.1, 0.10]),
                                    ("Linear", "Log"))
        self.b_new = Button(f.add_axes([0.01, 0.31, 0.1, 0.05]), "New wire")
        self.b_clear = Button(f.add_axes([0.01, 0.24, 0.1, 0.05]), "Clear all")

        def slider(y, label, lo, hi, init, **kw):
            return Slider(f.add_axes([0.2, y, 0.6, 0.025]), label, lo, hi,
                          valinit=init, **kw)
        self.s_I = slider(0.22, "Current (A)", 0.1, 10.0, 1.0)
        self.s_f = slider(0.18, "Frequency (Hz)", 0.1, 5.0, 0.5)
        self.s_h = slider(0.14, "Probe height (m)", 0.001, 10.0, 0.5)
        self.s_R = slider(0.10, "Axis Dimensions (m)", 0.05, 20.0, 5.0)
        self.s_seg = slider(0.06, "Max segment (mm)", 1, 100, 10, valstep=1)

    def _connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_click)
        canvas.mpl_connect("motion_notify_event", self.on_move)
        canvas.mpl_connect("key_press_event", self.on_key)
        self.b_new.on_clicked(lambda _: self.new_wire())
        self.b_clear.on_clicked(lambda _: self.clear())
        self.r_mode.on_clicked(
            lambda label: self.s_f.ax.set_visible(label == "AC"))
        # Anything that changes the 1 A field marks it for recomputation.
        for s in (self.s_h, self.s_R, self.s_seg):
            s.on_changed(lambda _: setattr(self, "dirty", True))

    # ------------------------------------------------------- user input
    def snap(self, x, y):
        """Round (x, y) to the nearest point of the selected snap grid."""
        s = SNAPS[self.r_snap.value_selected]
        return float(np.round(x / s) * s), float(np.round(y / s) * s)

    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        if ev.button == 1:
            self.wires[-1].append(self.snap(ev.xdata, ev.ydata))
            self.dirty = True
        elif ev.button == 3:
            self.undo()
        self.draw_wires()

    def on_move(self, ev):
        inside = ev.inaxes is self.ax and ev.xdata is not None
        self.cursor = self.snap(ev.xdata, ev.ydata) if inside else None
        self.draw_wires()

    def on_key(self, ev):
        if ev.key == "u":
            self.undo()
        elif ev.key == "n":
            self.new_wire()
        elif ev.key == "c":
            self.clear()
        self.draw_wires()

    def undo(self):
        """Remove the last vertex of the active wire."""
        if self.wires[-1]:
            self.wires[-1].pop()
            self.dirty = True

    def new_wire(self):
        """Close the active wire (if it has a vertex) and start another."""
        if self.wires[-1]:
            self.wires.append([])
        self.draw_wires()

    def clear(self):
        self.wires = [[]]
        self.dirty = True
        self.draw_wires()

    # ------------------------------------------------------ static drawing
    def draw_wires(self):
        """Redraw wires, the preview line, and the cursor."""
        xs, ys = [], []
        for w in self.wires:
            xs += [p[0] for p in w] + [np.nan]
            ys += [p[1] for p in w] + [np.nan]
        self.wire_line.set_data(xs, ys)

        active = self.wires[-1]
        if active and self.cursor:
            self.preview.set_data([active[-1][0], self.cursor[0]],
                                  [active[-1][1], self.cursor[1]])
        else:
            self.preview.set_data([], [])
        self.cursor_dot.set_data(*(([self.cursor[0]], [self.cursor[1]])
                                   if self.cursor else ([], [])))
        self.fig.canvas.draw_idle()

    def recompute(self):
        """Rebuild the 1 A field grid and flow paths from the current wires. This
        is the most computationally heavy portion and has room for improvement."""
        half = self.s_R.val
        xs = np.linspace(-half, half, N_EVAL)
        X, Y = np.meshgrid(xs, xs)
        self.grid_x, self.grid_y = X.ravel(), Y.ravel()

        # Update view: image extent, axis limits, arrow anchor points.
        self.im.set_extent([-half, half, -half, half])
        self.ax.set_xlim(-half, half)
        self.ax.set_ylim(-half, half)
        self.quiver.set_offsets(np.column_stack([self.grid_x, self.grid_y]))

        wires = [np.array(w, float) for w in self.wires if len(w) >= 2]
        total = sum(np.hypot(*np.diff(w, axis=0).T).sum() for w in wires)
        seg = max(self.s_seg.val / 1000.0, total / MAX_SEGS)

        mids, dls, self.paths = [], [], []
        for w in wires:
            m, d = splitwires(w, seg)
            mids.append(m)
            dls.append(d)
            cumulative = np.concatenate([[0.0],
                                  np.cumsum(np.hypot(*np.diff(w, axis=0).T))])
            if cumulative[-1] > 0:
                self.paths.append((w, cumulative))

        if mids:
            B = field_per_amp(self.grid_x, self.grid_y, self.s_h.val,
                              np.vstack(mids), np.vstack(dls))
        else:
            B = (np.zeros_like(self.grid_x),) * 3
        self.B = np.array(B).reshape(3, N_EVAL, N_EVAL)
        self.dirty = False

    # ------------------------------------------------- fast step (per frame)
    def render_field(self, I):
        """Colour and Length = |B| µT; arrows = in-plane direction"""
        Bx, By, Bz = I * self.B
        mag = np.sqrt(Bx**2 + By**2 + Bz**2) * 1e6           # µT

        # Colour limits: min and max of the field currently on screen.
        log = self.r_scale.value_selected == "Log"
        lo, hi = float(mag.min()), float(mag.max())
        if log:
            lo = max(lo, 1e-9)                    # LogNorm needs lo > 0
        hi = max(hi, lo * (1 + 1e-6), lo + 1e-12)  # avoid a zero-width range (e.g. I = 0)

        if log != self.log_scale:                 # swap norm only on toggle
            self.im.set_norm(LogNorm(lo, hi) if log else Normalize(lo, hi))
            self.cbar.set_label("|B| (µT, log)" if log else "|B| (µT)")
            self.cbar.update_normal(self.im)
            self.log_scale = log
        self.im.set_data(mag)
        self.im.set_clim(lo, hi)

        # Arrow length uses the same mapping as the colour scale (0..1).
        if log:
            frac = np.log(np.maximum(mag, lo) / lo) / np.log(hi / lo)
        else:
            frac = (mag - lo) / (hi - lo)
        frac = np.clip(frac, 0.0, 1.0)

        cell = 2 * self.s_R.val / (N_EVAL - 1)
        length = (0.15 + 0.85 * frac) * 0.9 * cell    # 0.15 floor keeps direction visible
        inplane = np.maximum(np.hypot(Bx, By), 1e-30)
        self.quiver.set_UVC((Bx / inplane * length).ravel(),
                            (By / inplane * length).ravel())

    def render_flow(self, I, amp):
        """This is the function handling the dots representing moving current"""
        self.phase = (self.phase + DOT_SPEED * I / amp) % 1.0
        xs, ys = [], []
        for w, cum in self.paths:
            s = ((np.arange(N_DOTS) / N_DOTS + self.phase) % 1.0) * cum[-1]
            xs.append(np.interp(s, cum, w[:, 0]))     # position by arc length
            ys.append(np.interp(s, cum, w[:, 1]))
        if xs:
            self.flow.set_data(np.concatenate(xs), np.concatenate(ys))
        else:
            self.flow.set_data([], [])
        self.flow.set_color("yellow" if I >= 0 else "cyan")

    def update(self, frame):
        if self.dirty:
            self.recompute()
        amp = self.s_I.val
        ac = self.r_mode.value_selected == "AC"
        I = amp * np.sin(2 * np.pi * self.s_f.val * frame * DT) if ac else amp
        self.render_field(I)
        self.render_flow(I, amp)
        n = sum(len(w) >= 2 for w in self.wires)
        self.title.set_text(f"{'AC' if ac else 'DC'}: I = {I:+.2f} A   {n} wire(s)   "
                            f"n: new, u: undo, c: clear")
        return (self.im, self.quiver, self.flow, self.title,
                self.wire_line, self.preview, self.cursor_dot)


if __name__ == "__main__":
    sim = WireSim()
    sim.ani = FuncAnimation(sim.fig, sim.update, interval=int(DT * 1000),cache_frame_data=False, blit=True)
    #Note that blit makes the grid flash white for reasons i don't know, but it improves the framerate enough to tolerate
    plt.show()