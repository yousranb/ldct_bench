import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
import prepare_nifti as pn
from register import register_to_reference

root, subj, name = sys.argv[1], sys.argv[2], sys.argv[3]
mesh = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
pn.OPT.update(stride=2)
ref, _, sp = pn.load_nifti(pn.series_path(root, subj, "normal"))
mov, _, _ = pn.load_nifti(pn.series_path(root, subj, name))
w, rep = register_to_reference(ref, mov, (sp[2], sp[0], sp[1]), mesh_mm=mesh)
body = np.array([(s[::8, ::8] > -500).mean() for s in ref])
cand = np.where(body > 0.3)[0]
idx = cand[np.linspace(0, len(cand) - 1, 5).astype(int)[1:4]]
lp = lambda x: ndimage.gaussian_filter(x, 3)
titles = ["normal", name + " (stored)", name + " (registered)", "low-pass diff before", "low-pass diff after"]
fig, ax = plt.subplots(3, 5, figsize=(17, 10.5))
for r, i in enumerate(idx):
    ims = [ref[i], mov[i], w[i], lp(mov[i]) - lp(ref[i]), lp(w[i]) - lp(ref[i])]
    for c, im in enumerate(ims):
        kw = dict(cmap="gray", vmin=-160, vmax=240) if c < 3 else dict(cmap="RdBu_r", vmin=-150, vmax=150)
        ax[r, c].imshow(im, **kw); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(titles[c])
    ax[r, 0].set_ylabel(f"slice {i}")
fig.suptitle(f"{subj} {name}: low-pass diff {rep['lp_before']:.1f} -> {rep['lp_after']:.1f} HU (mesh {mesh:g} mm)")
fig.tight_layout(); out = f"qc_{subj}_{name}.png"; fig.savefig(out, dpi=70); print("saved", out)
