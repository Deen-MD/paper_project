"""
AI-Based Detection of Illegal Mining Activities
Using Satellite Imagery and Machine Learning
------------------------------------------------
A runnable reference implementation of the pipeline described in the paper
presentation:

    DATA -> PREPROCESS -> AI/ML SEGMENTATION -> BITEMPORAL CHANGE DETECTION
         -> GIS LEASE-BOUNDARY VERIFICATION -> CONFIDENCE SCORING
         -> DASHBOARD / ALERT REPORT (human-in-the-loop review)

The script supports a synthetic demo and live Sentinel-2 imagery from
Google Earth Engine. Live imagery requires an Earth Engine account, an
enabled Cloud project, and authentication. Before the first live run, use
`python -c "import ee; ee.Authenticate()"` in the project environment.

Design principle carried over from the slides: the model output is a
*review candidate*, not a legal conclusion. Every flagged site is routed
to a human reviewer with geo-coordinates, a confidence score, and
before/after evidence.

Dependencies: numpy, scipy, shapely, earthengine-api
Optional:     torch (only needed for the real trainable U-Net; the
              pipeline falls back to a spectral-index segmenter if torch
              is not installed).
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from urllib.request import urlopen

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, box, mapping

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ======================================================================
# 1. CONFIG
# ======================================================================

@dataclass
class PipelineConfig:
    patch_size: int = 128                 # pixels, square patch
    pixel_resolution_m: float = 10.0      # Sentinel-2-like GSD
    min_change_area_px: int = 25          # spatial/minimum-area filter
    change_threshold: float = 0.35        # bitemporal difference threshold
    mining_class_id: int = 3              # land-cover class index for "mining/barren-disturbed"
    confidence_threshold: float = 0.5     # below this, don't raise an alert


LAND_COVER_CLASSES = {0: "water", 1: "vegetation", 2: "built-up/agriculture", 3: "barren/mining"}


# ======================================================================
# 2. DATA ACQUISITION  ("OBSERVE" in the abstract slide)
# ======================================================================

class SyntheticImageSource:
    """
    Stands in for a Sentinel-2 / Landsat / drone loader.
    Produces a pair of time-stamped multispectral patches (T1, T2) with
    a simulated new disturbance (mining/excavation scar) appearing at T2,
    plus a lat/lon origin so results can be geo-tagged.
    """

    def __init__(self, config: PipelineConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.default_rng(seed)

    def _base_scene(self) -> np.ndarray:
        """4-band synthetic scene: [R, G, B, NIR], values in reflectance [0,1]."""
        n = self.cfg.patch_size
        scene = np.zeros((4, n, n), dtype=np.float32)
        # vegetation-dominated background: low R, moderate G, low B, high NIR
        scene[0] = self.rng.normal(0.10, 0.02, (n, n))
        scene[1] = self.rng.normal(0.15, 0.02, (n, n))
        scene[2] = self.rng.normal(0.08, 0.02, (n, n))
        scene[3] = self.rng.normal(0.45, 0.05, (n, n))
        # a river (water: low across all, very low NIR)
        scene[:, n // 2 - 3: n // 2 + 3, :] = np.array([0.05, 0.07, 0.09, 0.03])[:, None, None]
        return np.clip(scene, 0, 1)

    def _add_disturbance(self, scene: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """Bare/excavated soil: high R/G/B, low NIR -- spectrally distinct from vegetation."""
        scene = scene.copy()
        n = scene.shape[-1]
        yy, xx = np.ogrid[:n, :n]
        mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
        scene[0][mask] = self.rng.normal(0.55, 0.03, mask.sum())
        scene[1][mask] = self.rng.normal(0.50, 0.03, mask.sum())
        scene[2][mask] = self.rng.normal(0.45, 0.03, mask.sum())
        scene[3][mask] = self.rng.normal(0.15, 0.03, mask.sum())
        return np.clip(scene, 0, 1)

    def get_pair(self, lease_status: str = "expired") -> Dict:
        """
        Returns T1 (baseline) and T2 (recent) imagery for one site, plus
        metadata. `lease_status` controls where the simulated disturbance
        falls relative to the lease polygon, so the GIS-check stage has
        something meaningful to test.
        """
        n = self.cfg.patch_size
        t1 = self._base_scene()

        if lease_status == "active":
            # disturbance inside an active, authorized lease -> should NOT alert
            center = (n // 2, n // 4)
        else:
            # disturbance outside / on expired lease -> SHOULD alert
            center = (n // 4, 3 * n // 4)

        t2 = self._add_disturbance(t1, center=center, radius=14)

        origin_lat, origin_lon = 21.1702, 79.0871  # arbitrary demo coordinates
        return {
            "site_id": f"SITE-{self.rng.integers(1000, 9999)}",
            "t1": t1,
            "t2": t2,
            "origin": (origin_lat, origin_lon),
            "lease_status": lease_status,
            "disturbance_center_px": center,
        }


class EarthEngineImageSource:
    """Downloads a small, cloud-masked Sentinel-2 pair from Earth Engine."""

    bands = ("B4", "B3", "B2", "B8")
    collection_id = "COPERNICUS/S2_SR_HARMONIZED"
    max_scene_cloud_percent = 30
    lookback_days = 365
    min_pair_gap_days = 30

    def __init__(self, config: PipelineConfig, project_id: str):
        try:
            import ee
        except ImportError as exc:
            raise RuntimeError(
                "Google Earth Engine support is not installed. "
                "Install dependencies with: "
                "python -m pip install -r paper/requirements.txt"
            ) from exc

        self.cfg = config
        self.ee = ee
        ee.Initialize(project=project_id)

    def _image_for_time(self, collection, timestamp_ms: int):
        ee = self.ee
        images = collection.filter(
            ee.Filter.eq("system:time_start", timestamp_ms)
        ).sort("CLOUDY_PIXEL_PERCENTAGE", opt_ascending=False)
        mosaic = ee.Image(images.mosaic())

        scl = mosaic.select("SCL")
        clear = scl.gte(2).And(scl.lte(6))
        reflectance = (
            mosaic.select(list(self.bands))
            .updateMask(clear)
            .multiply(0.0001)
            .unmask(-9999)
        )
        return reflectance.addBands(clear.unmask(0).rename("valid"))

    def _download_patch(self, image, region) -> Tuple[np.ndarray, np.ndarray]:
        bands = [*self.bands, "valid"]
        url = image.getDownloadURL({
            "bands": bands,
            "region": region,
            "dimensions": [self.cfg.patch_size, self.cfg.patch_size],
            "format": "NPY",
        })
        with urlopen(url, timeout=90) as response:
            data = np.load(io.BytesIO(response.read()), allow_pickle=False)

        if data.dtype.names is None or not set(bands).issubset(data.dtype.names):
            raise RuntimeError(
                "Earth Engine returned an unexpected NumPy image format."
            )

        image_bands = np.stack(
            [data[name] for name in self.bands], axis=0
        ).astype(np.float32)
        valid_mask = data["valid"] > 0
        expected_shape = (self.cfg.patch_size, self.cfg.patch_size)
        if image_bands.shape != (len(self.bands), *expected_shape):
            raise RuntimeError(
                f"Earth Engine returned image shape {image_bands.shape}; "
                f"expected {(len(self.bands), *expected_shape)}."
            )

        image_bands[:, ~valid_mask] = np.nan
        return image_bands, valid_mask

    def get_pair(
        self, latitude: float, longitude: float, width_km: float
    ) -> Dict:
        ee = self.ee
        region = ee.Geometry.Point([longitude, latitude]).buffer(
            width_km * 500
        ).bounds()
        now = datetime.now(timezone.utc)
        start = now.timestamp() - self.lookback_days * 24 * 60 * 60

        collection = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(region)
            .filterDate(
                datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d"),
                now.strftime("%Y-%m-%d"),
            )
            .filter(ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE", self.max_scene_cloud_percent
            ))
        )
        timestamps = collection.aggregate_array(
            "system:time_start"
        ).distinct().getInfo()
        timestamps = sorted(int(value) for value in timestamps)
        if len(timestamps) < 2:
            raise RuntimeError(
                "Earth Engine found fewer than two suitable Sentinel-2 "
                "acquisitions in the past year for this location. Try another "
                "location or a larger footprint."
            )

        recent_time = timestamps[-1]
        gap_ms = self.min_pair_gap_days * 24 * 60 * 60 * 1000
        earlier_times = [
            value for value in timestamps[:-1]
            if recent_time - value >= gap_ms
        ]
        if not earlier_times:
            raise RuntimeError(
                "Earth Engine found recent imagery, but no pair at least "
                f"{self.min_pair_gap_days} days apart. Try another location."
            )
        earlier_time = earlier_times[-1]

        t1, valid_t1 = self._download_patch(
            self._image_for_time(collection, earlier_time), region
        )
        t2, valid_t2 = self._download_patch(
            self._image_for_time(collection, recent_time), region
        )
        earlier_date = datetime.fromtimestamp(
            earlier_time / 1000, timezone.utc
        ).date().isoformat()
        recent_date = datetime.fromtimestamp(
            recent_time / 1000, timezone.utc
        ).date().isoformat()

        self.cfg.pixel_resolution_m = width_km * 1000 / self.cfg.patch_size
        return {
            "site_id": f"S2-{latitude:.3f}-{longitude:.3f}",
            "t1": t1,
            "t2": t2,
            "valid_mask_t1": valid_t1,
            "valid_mask_t2": valid_t2,
            "origin": (latitude, longitude),
            "acquisition_dates": [earlier_date, recent_date],
            "source": "Sentinel-2 SR Harmonized via Google Earth Engine",
        }


# ======================================================================
# 3. PREPROCESSING  (cloud mask, normalization, co-registration)
# ======================================================================

class Preprocessor:
    def cloud_mask(self, img: np.ndarray, brightness_thresh: float = 0.9) -> np.ndarray:
        """Very bright, spectrally flat pixels across all bands ~ clouds."""
        mean_reflectance = img.mean(axis=0)
        flatness = img.std(axis=0)
        cloud = (mean_reflectance > brightness_thresh) & (flatness < 0.03)
        return cloud  # boolean mask, True = cloud

    def normalize(self, img: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Per-band min-max normalization to [0, 1]."""
        out = np.empty_like(img)
        for b in range(img.shape[0]):
            band = img[b]
            valid = valid_mask & np.isfinite(band)
            if not valid.any():
                raise ValueError("An input image has no valid pixels to normalize.")
            lo, hi = np.percentile(band[valid], [1, 99])
            out[b] = 0
            out[b][valid] = np.clip(
                (band[valid] - lo) / max(hi - lo, 1e-6), 0, 1
            )
        return out

    def co_register(
        self, t1: np.ndarray, t2: np.ndarray, max_shift: int = 4
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Cross-correlation based alignment: finds the integer (dy, dx) shift
        of t2 that best matches t1 and rolls t2 back into alignment.
        A lightweight stand-in for real sub-pixel co-registration.
        """
        ref = t1.mean(axis=0)
        best_shift, best_score = (0, 0), -np.inf
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                shifted = np.roll(t2.mean(axis=0), (dy, dx), axis=(0, 1))
                score = np.corrcoef(ref.ravel(), shifted.ravel())[0, 1]
                if score > best_score:
                    best_score, best_shift = score, (dy, dx)
        return np.roll(t2, best_shift, axis=(1, 2)), best_shift

    def run(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        valid_t1: Optional[np.ndarray] = None,
        valid_t2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
        finite_t1 = np.isfinite(t1).all(axis=0)
        finite_t2 = np.isfinite(t2).all(axis=0)
        valid_t1_mask = (
            finite_t1 if valid_t1 is None else np.asarray(valid_t1, dtype=bool) & finite_t1
        )
        valid_t2_mask = (
            finite_t2 if valid_t2 is None else np.asarray(valid_t2, dtype=bool) & finite_t2
        )
        cloud1, cloud2 = self.cloud_mask(t1), self.cloud_mask(t2)
        t1n = self.normalize(t1, valid_t1_mask)
        t2n = self.normalize(t2, valid_t2_mask)
        cloud_fraction_t1 = float(
            (cloud1 & valid_t1_mask).sum() / max(valid_t1_mask.sum(), 1)
        )
        cloud_fraction_t2 = float(
            (cloud2 & valid_t2_mask).sum() / max(valid_t2_mask.sum(), 1)
        )
        t2n, shift = self.co_register(t1n, t2n)
        valid_t2_mask = np.roll(valid_t2_mask, shift, axis=(0, 1))
        valid = valid_t1_mask & valid_t2_mask
        info = {
            "cloud_fraction_t1": cloud_fraction_t1,
            "cloud_fraction_t2": cloud_fraction_t2,
            "valid_fraction": float(valid.mean()),
        }
        return t1n, t2n, info, valid


# ======================================================================
# 4. AI/ML SEGMENTATION  (U-Net / ResNet in the slides)
# ======================================================================

if TORCH_AVAILABLE:
    class UNet(nn.Module):
        """
        Compact U-Net for 4-band (R,G,B,NIR) land-cover segmentation into
        {water, vegetation, built-up/agriculture, barren/mining}.
        This matches the "U-Net / ResNet -- Segmentation" block in the
        architecture slide. Untrained here -- plug in your own weights
        (`model.load_state_dict(...)`) once trained on labeled imagery.
        """

        def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 16):
            super().__init__()

            def conv_block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                )

            self.enc1 = conv_block(in_channels, base)
            self.enc2 = conv_block(base, base * 2)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = conv_block(base * 2, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = conv_block(base * 2, base)
            self.out_conv = nn.Conv2d(base, num_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b = self.bottleneck(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out_conv(d1)  # logits: (B, num_classes, H, W)


class SpectralIndexSegmenter:
    """
    Deterministic fallback segmenter used whenever no trained model is
    available. It classifies each pixel from NDVI/NDWI-style spectral indices
    computed from R, G, B, NIR -- no training data required.
    """

    def segment(self, img: np.ndarray) -> np.ndarray:
        r, g, b, nir = img[0], img[1], img[2], img[3]
        ndvi = (nir - r) / (nir + r + 1e-6)     # vegetation index
        ndwi = (g - nir) / (g + nir + 1e-6)     # water index
        brightness = (r + g + b) / 3.0

        labels = np.full(r.shape, LAND_COVER_CLASS_LOOKUP["vegetation"], dtype=np.int32)

        # Water: low reflectance and a clear green-vs-NIR signature.
        water_mask = (ndwi > 0.20) & (brightness < 0.35)
        labels[water_mask] = LAND_COVER_CLASS_LOOKUP["water"]

        # Bare land / mining disturbance: bright soil with low NIR and weak vegetation.
        mining_mask = (ndvi < 0.15) & (brightness > 0.35)
        labels[mining_mask] = LAND_COVER_CLASS_LOOKUP["barren/mining"]

        # Built-up / agricultural surfaces: low-to-moderate vegetation but not bare soil.
        built_mask = (ndvi >= 0.15) & (ndvi < 0.35)
        labels[built_mask] = LAND_COVER_CLASS_LOOKUP["built-up/agriculture"]
        return labels


LAND_COVER_CLASS_LOOKUP = {v: k for k, v in LAND_COVER_CLASSES.items()}


class Segmenter:
    """Prefer the spectral fallback unless a trained model is explicitly supplied."""

    def __init__(self, config: PipelineConfig, model: Optional[nn.Module] = None):
        self.cfg = config
        self.fallback = SpectralIndexSegmenter()
        self.model = model if model is not None else None
        if self.model is not None:
            self.model.eval()

    def segment(self, img: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self.fallback.segment(img)
        with torch.no_grad():
            x = torch.from_numpy(img).unsqueeze(0).float()
            logits = self.model(x)
            return logits.argmax(dim=1).squeeze(0).numpy()


# ======================================================================
# 5. BITEMPORAL CHANGE DETECTION  (Siamese-style comparison)
# ======================================================================

class ChangeDetector:
    """
    Combines (a) raw spectral difference between T1/T2 and (b) land-cover
    class transitions into "barren/mining" to flag newly disturbed pixels.
    This mirrors the "Siamese network / pixel-wise difference" block.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def detect(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        spectral_diff = np.linalg.norm(t2 - t1, axis=0)  # per-pixel L2 change magnitude
        spectral_diff /= spectral_diff.max() + 1e-6

        became_mining = (labels2 == self.cfg.mining_class_id) & (labels1 != self.cfg.mining_class_id)
        change_mask = became_mining & (spectral_diff > self.cfg.change_threshold)
        if valid_mask is not None:
            change_mask &= valid_mask

        # spatial/minimum-area filter: drop tiny/noisy blobs
        labeled_blobs, n_blobs = ndimage.label(change_mask)
        sizes = ndimage.sum(change_mask, labeled_blobs, range(1, n_blobs + 1))
        keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= self.cfg.min_change_area_px]
        filtered_mask = np.isin(labeled_blobs, keep_ids)

        return {
            "spectral_diff": spectral_diff,
            "raw_change_mask": change_mask,
            "filtered_change_mask": filtered_mask,
            "blob_labels": labeled_blobs,
            "kept_blob_ids": keep_ids,
        }


# ======================================================================
# 6. GIS LEASE-BOUNDARY VERIFICATION
# ======================================================================

@dataclass
class MiningLease:
    lease_id: str
    polygon_px: Polygon       # boundary in local pixel/site coordinates for this demo
    status: str                # "active" | "expired" | "revoked"


class LeaseRegistry:
    """Stands in for a PostGIS/QGIS lease-boundary layer."""

    def __init__(self, leases: List[MiningLease]):
        self.leases = leases

    def check(self, point_px: Tuple[float, float]) -> Optional[MiningLease]:
        from shapely.geometry import Point
        pt = Point(point_px[1], point_px[0])  # (x=col, y=row)
        for lease in self.leases:
            if lease.polygon_px.contains(pt):
                return lease
        return None

    def is_authorized(self, point_px: Tuple[float, float]) -> bool:
        lease = self.check(point_px)
        return lease is not None and lease.status == "active"


def build_demo_lease_registry(patch_size: int, active_center: Tuple[int, int]) -> LeaseRegistry:
    """Builds one active lease around `active_center`; everywhere else is unauthorized."""
    r, c = active_center
    active_poly = box(c - 20, r - 20, c + 20, r + 20)
    return LeaseRegistry([MiningLease("LEASE-A1", active_poly, "active")])


# ======================================================================
# 7. CONFIDENCE SCORING
# ======================================================================

class ConfidenceScorer:
    """
    Confidence = f(change magnitude, blob size, class-transition strength).
    A simple, explainable combination -- intentionally not a black box,
    since the design principle is "AI assists review, doesn't decide".
    """

    def score(self, spectral_diff: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        magnitude_term = float(spectral_diff[mask].mean())          # 0..1
        size_term = min(1.0, mask.sum() / 200.0)                     # saturates at 200 px
        confidence = 0.6 * magnitude_term + 0.4 * size_term
        return round(float(np.clip(confidence, 0, 1)), 3)


# ======================================================================
# 8. REPORTING / DASHBOARD / ALERTING
# ======================================================================

class AlertReporter:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def build_case(self, site: Dict, blob_id: int, blob_labels: np.ndarray,
                    confidence: float, lease: Optional[MiningLease]) -> Dict:
        ys, xs = np.where(blob_labels == blob_id)
        centroid_px = (float(ys.mean()), float(xs.mean()))
        area_px = int(len(ys))
        area_m2 = area_px * (self.cfg.pixel_resolution_m ** 2)

        lat, lon = site["origin"]
        # crude local-degrees offset for the demo (not geodetically precise)
        deg_per_px = self.cfg.pixel_resolution_m / 111_320.0
        n = self.cfg.patch_size
        geo_lat = lat + (centroid_px[0] - n / 2) * deg_per_px
        geo_lon = lon + (centroid_px[1] - n / 2) * deg_per_px

        return {
            "site_id": site["site_id"],
            "centroid_px": centroid_px,
            "centroid_latlon": [round(geo_lat, 6), round(geo_lon, 6)],
            "area_px": area_px,
            "area_m2": area_m2,
            "confidence": confidence,
            "lease_status_at_location": lease.status if lease else "no_registered_lease",
            "lease_id": lease.lease_id if lease else None,
            # Candidate unauthorized site = disturbance outside an active authorized
            # boundary, or within a revoked/expired boundary (per the problem statement).
            "classification": (
                "candidate_unauthorized_site"
                if confidence >= self.cfg.confidence_threshold and (lease is None or lease.status != "active")
                else ("authorized_activity" if lease is not None and lease.status == "active" else "below_threshold")
            ),
            "note": "AI-flagged review candidate -- not a legal determination. Requires human-in-the-loop verification.",
        }

    def to_geojson(self, cases: List[Dict]) -> Dict:
        features = []
        for c in cases:
            lat, lon = c["centroid_latlon"]
            features.append({
                "type": "Feature",
                "geometry": mapping(Polygon([  # tiny square marker around the point
                    (lon - 0.0002, lat - 0.0002), (lon + 0.0002, lat - 0.0002),
                    (lon + 0.0002, lat + 0.0002), (lon - 0.0002, lat + 0.0002),
                ])),
                "properties": {k: v for k, v in c.items() if k not in ("centroid_px",)},
            })
        return {"type": "FeatureCollection", "features": features}


# ======================================================================
# 9. END-TO-END PIPELINE
# ======================================================================

class IllegalMiningDetectionPipeline:
    """
    OBSERVE -> DETECT -> VERIFY -> REPORT
    Wires every stage above into the workflow shown in the architecture slide.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.preprocessor = Preprocessor()
        self.segmenter = Segmenter(self.cfg)
        self.change_detector = ChangeDetector(self.cfg)
        self.scorer = ConfidenceScorer()
        self.reporter = AlertReporter(self.cfg)

    def run_on_site(self, site: Dict, lease_registry: LeaseRegistry) -> List[Dict]:
        # 1. PREPROCESS
        t1n, t2n, cloud_info, valid_mask = self.preprocessor.run(
            site["t1"],
            site["t2"],
            site.get("valid_mask_t1"),
            site.get("valid_mask_t2"),
        )

        # 2. AI/ML SEGMENTATION
        labels1 = self.segmenter.segment(t1n)
        labels2 = self.segmenter.segment(t2n)

        # 3. BITEMPORAL CHANGE DETECTION (+ spatial/min-area filter)
        change = self.change_detector.detect(
            t1n, t2n, labels1, labels2, valid_mask
        )

        # 4. GIS LEASE-BOUNDARY VERIFICATION + 5. CONFIDENCE SCORING + 6. REPORT
        cases = []
        for blob_id in change["kept_blob_ids"]:
            blob_mask = change["blob_labels"] == blob_id
            confidence = self.scorer.score(change["spectral_diff"], blob_mask)

            ys, xs = np.where(blob_mask)
            centroid_px = (float(ys.mean()), float(xs.mean()))
            lease = lease_registry.check(centroid_px)

            case = self.reporter.build_case(site, blob_id, change["blob_labels"], confidence, lease)
            case["cloud_info"] = cloud_info
            cases.append(case)

        return cases


# ======================================================================
# 10. DEMO RUN
# ======================================================================

def _write_reports(pipeline, all_cases: List[Dict], total_sites: int) -> None:
    flagged = [
        case for case in all_cases
        if case["classification"] == "candidate_unauthorized_site"
    ]
    summary = {
        "total_sites_processed": total_sites,
        "candidate_unauthorized_sites": len(flagged),
        "avg_confidence_flagged": (
            round(float(np.mean([case["confidence"] for case in flagged])), 3)
            if flagged else 0.0
        ),
    }
    print("Dashboard summary:", json.dumps(summary, indent=2))
    geojson = pipeline.reporter.to_geojson(all_cases)
    with open("mining_alerts.geojson", "w") as file:
        json.dump(geojson, file, indent=2)
    with open("mining_report.json", "w") as file:
        json.dump(all_cases, file, indent=2)
    print("\nWrote mining_alerts.geojson and mining_report.json")


def _print_cases(cases: List[Dict]) -> None:
    if not cases:
        print("  No disturbance flagged.\n")
        return
    for case in cases:
        print(
            f"  Candidate site  | area: {case['area_m2']:.0f} m^2"
            f"  | confidence: {case['confidence']:.2f}"
            f"  | lease status at location: {case['lease_status_at_location']}"
            f"  | classification: {case['classification']}"
        )
    print()


def _run_synthetic_demo() -> None:
    cfg = PipelineConfig()
    pipeline = IllegalMiningDetectionPipeline(cfg)
    source = SyntheticImageSource(cfg)

    model_label = (
        "U-Net (PyTorch, trained weights required)"
        if pipeline.segmenter.model is not None
        else "Spectral-index fallback (deterministic demo mode)"
    )
    print(f"Backend segmentation model: {model_label}\n")

    all_cases = []
    for lease_status in ("expired", "active"):
        site = source.get_pair(lease_status=lease_status)
        lease_registry = build_demo_lease_registry(
            cfg.patch_size,
            active_center=(cfg.patch_size // 2, cfg.patch_size // 4),
        )
        cases = pipeline.run_on_site(site, lease_registry)
        all_cases.extend(cases)

        print(f"--- Site {site['site_id']}  (simulated lease context: {lease_status}) ---")
        _print_cases(cases)

    _write_reports(pipeline, all_cases, total_sites=2)


def _run_earth_engine() -> None:
    try:
        location = input("Enter center latitude, longitude (example: 21.17, 79.09): ")
        latitude, longitude = (float(value.strip()) for value in location.split(","))
    except ValueError as exc:
        raise ValueError(
            "Enter the location as two numbers separated by a comma: latitude, longitude."
        ) from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Latitude must be in [-90, 90] and longitude in [-180, 180].")

    width_entry = input("Image footprint width in km [1.28]: ").strip()
    width_km = float(width_entry) if width_entry else 1.28
    if width_km <= 0:
        raise ValueError("The image footprint width must be greater than zero.")

    project_id = os.environ.get("EARTHENGINE_PROJECT", "").strip()
    if not project_id:
        project_id = input("Google Earth Engine Cloud project ID: ").strip()
    if not project_id:
        raise ValueError(
            "A Google Cloud project ID is required. Set EARTHENGINE_PROJECT "
            "or enter the project ID when prompted."
        )

    cfg = PipelineConfig()
    source = EarthEngineImageSource(cfg, project_id)
    site = source.get_pair(latitude, longitude, width_km)
    pipeline = IllegalMiningDetectionPipeline(cfg)

    print(f"\nImagery source: {site['source']}")
    print(f"Acquisition dates (T1, T2): {site['acquisition_dates']}")
    print(
        "No real mining-lease boundaries are loaded. Any detection is an "
        "unverified review candidate, not a legal determination.\n"
    )
    cases = pipeline.run_on_site(site, LeaseRegistry([]))
    print(f"--- Site {site['site_id']} ---")
    _print_cases(cases)
    _write_reports(pipeline, cases, total_sites=1)


def main():
    mode = input(
        "Imagery source: [G]oogle Earth Engine (live) or "
        "[S]ynthetic demo (default: G): "
    ).strip().lower() or "g"
    if mode in {"s", "synthetic"}:
        _run_synthetic_demo()
    elif mode in {"g", "google", "earth engine"}:
        _run_earth_engine()
    else:
        raise ValueError("Choose G for Google Earth Engine or S for the synthetic demo.")


if __name__ == "__main__":
    main()