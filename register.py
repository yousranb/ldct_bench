"""Deformable registration of a low-dose volume onto the normal-dose volume (SimpleITK).

Why this direction: the NDCT is the network input and the reference grid; the LDCT is
resampled onto it with NEAREST-NEIGHBOUR interpolation, so that its voxels keep their
original noise values (linear/B-spline interpolation would average neighbouring voxels and
reduce the noise variance, i.e. falsify the NPS we want to learn and to evaluate).

    from register import register_to_reference
    warped, report = register_to_reference(ndct, ldct, spacing_slices_rows_cols)
"""
import numpy as np
from scipy import ndimage


def lowpass_diff_std(a, b, idx, s=3.0):
    """median over slices `idx` of std(G_s(b) - G_s(a)) in the body: ~0 when aligned."""
    return float(np.median([np.std((ndimage.gaussian_filter(b[i], s) - ndimage.gaussian_filter(a[i], s))[a[i] > -500])
                            for i in idx]))


def register_to_reference(ref, mov, spacing, mesh_mm=50.0, shrink=(4, 2), iters=60, sampling=0.02,
                          verbose=True):
    """ref, mov: (slices, rows, cols) HU arrays on the same grid; spacing = (slice_mm, row_mm, col_mm).
    Returns (mov warped onto ref, dict with low-pass diff std before/after)."""
    import SimpleITK as sitk

    def img(x, clip=True):
        x = np.clip(x, -1000, 1500) if clip else x
        im = sitk.GetImageFromArray(np.ascontiguousarray(x, np.float32))
        im.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))   # (x=cols, y=rows, z=slices)
        return im

    F, M = img(ref), img(mov)
    size_mm = [sz * sp for sz, sp in zip(F.GetSize(), F.GetSpacing())]
    mesh = [max(1, int(round(s / mesh_mm))) for s in size_mm]
    tx = sitk.BSplineTransformInitializer(F, mesh)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(sampling, 0)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iters,
                           maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=1000,
                           costFunctionConvergenceFactor=1e7)
    R.SetInitialTransform(tx, inPlace=True)
    R.SetShrinkFactorsPerLevel(list(shrink))
    R.SetSmoothingSigmasPerLevel([2.0, 1.0][:len(shrink)] + [0.0] * max(0, len(shrink) - 2))
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    R.Execute(F, M)
    W = sitk.Resample(img(mov, clip=False), F, tx, sitk.sitkNearestNeighbor, -1024.0)
    warped = sitk.GetArrayFromImage(W).astype(np.float32)

    body = np.array([(s[::8, ::8] > -500).mean() for s in ref])
    cand = np.where(body > 0.3)[0]
    idx = cand[np.linspace(0, len(cand) - 1, min(12, len(cand))).astype(int)] if len(cand) else [len(ref) // 2]
    rep = dict(lp_before=lowpass_diff_std(ref, mov, idx), lp_after=lowpass_diff_std(ref, warped, idx),
               mesh=mesh, metric=float(R.GetMetricValue()))
    if verbose:
        print(f"      registration: B-spline mesh {mesh}, low-pass diff std "
              f"{rep['lp_before']:.1f} -> {rep['lp_after']:.1f} HU")
    return warped, rep
