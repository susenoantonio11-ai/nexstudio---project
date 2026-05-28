"""
GeoServer Integration untuk WMS/WMTS Serving
============================================
Wrapper untuk GeoServer REST API agar Nexlytics bisa publish hasil flood mapping
sebagai web map service yang bisa di-load di Leaflet/Mapbox/QGIS.

Workflow:
1. Setup GeoServer instance (Docker recommended)
2. Create workspace untuk project
3. Create coverage store (untuk raster) atau datastore (untuk vector)
4. Publish layer dari GeoTIFF/Shapefile
5. Apply SLD style (color ramp untuk flood probability, dll)
6. Akses via WMS: http://geoserver/geoserver/{workspace}/wms
   atau WMTS: http://geoserver/geoserver/gwc/service/wmts

Setup GeoServer cepat dengan Docker:
    docker run -p 8080:8080 -e GEOSERVER_DATA_DIR=/data \
      -v $(pwd)/geoserver-data:/data \
      docker.osgeo.org/geoserver:latest
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os


def _try_requests():
    try:
        import requests
        return requests
    except ImportError:
        return None


_REQUESTS = _try_requests()


class GeoServerClient:
    """
    REST API client untuk GeoServer.

    Args:
        base_url: e.g., 'http://localhost:8080/geoserver'
        username: GeoServer admin username (default 'admin')
        password: GeoServer admin password (default 'geoserver')
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/geoserver",
        username: str = "admin",
        password: str = "geoserver",
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = (username, password)
        self.rest_url = f"{self.base_url}/rest"

    def is_available(self) -> bool:
        return _REQUESTS is not None

    def health_check(self) -> Dict[str, Any]:
        """Cek apakah GeoServer reachable + auth valid."""
        if not self.is_available():
            return {
                "available": False,
                "reason": "requests library belum terinstall",
                "install": "pip install requests",
            }
        try:
            r = _REQUESTS.get(
                f"{self.rest_url}/about/version.json",
                auth=self.auth,
                timeout=5,
            )
            if r.status_code == 200:
                version_data = r.json()
                return {
                    "available": True,
                    "status": "online",
                    "geoserver_version": version_data.get("about", {}),
                    "rest_url": self.rest_url,
                }
            return {
                "available": False,
                "status_code": r.status_code,
                "error": "Auth failed atau GeoServer tidak terjangkau",
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "hint": (
                    "Pastikan GeoServer running. Quick start:\n"
                    "  docker run -p 8080:8080 docker.osgeo.org/geoserver:latest"
                ),
            }

    # ==================================================================
    # WORKSPACE MANAGEMENT
    # ==================================================================
    def create_workspace(self, name: str) -> Dict[str, Any]:
        """Create workspace (logical container untuk layers)."""
        if not self.is_available():
            return self._stub("create_workspace", name)
        try:
            r = _REQUESTS.post(
                f"{self.rest_url}/workspaces",
                json={"workspace": {"name": name}},
                auth=self.auth,
                timeout=10,
            )
            if r.status_code in (201, 409):
                return {
                    "success": r.status_code == 201,
                    "already_exists": r.status_code == 409,
                    "workspace_name": name,
                    "wms_url": f"{self.base_url}/{name}/wms",
                    "wmts_url": f"{self.base_url}/gwc/service/wmts",
                }
            return {"success": False, "status_code": r.status_code, "response": r.text[:300]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_workspaces(self) -> Dict[str, Any]:
        """List semua workspaces."""
        if not self.is_available():
            return self._stub("list_workspaces")
        try:
            r = _REQUESTS.get(
                f"{self.rest_url}/workspaces.json",
                auth=self.auth,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json().get("workspaces", {})
                workspaces = data.get("workspace", []) if isinstance(data, dict) else []
                return {
                    "success": True,
                    "n_workspaces": len(workspaces),
                    "workspaces": [w.get("name") for w in workspaces],
                }
            return {"success": False, "status_code": r.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================================================================
    # PUBLISH GEOTIFF AS COVERAGE
    # ==================================================================
    def publish_geotiff(
        self,
        workspace: str,
        layer_name: str,
        geotiff_path: str,
        coverage_store: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload + publish GeoTIFF sebagai coverage store + layer.

        Resulting WMS URL:
            http://geoserver/geoserver/{workspace}/wms?
              SERVICE=WMS&REQUEST=GetMap&LAYERS={workspace}:{layer_name}
              &BBOX=...&WIDTH=...&HEIGHT=...&CRS=EPSG:4326&FORMAT=image/png
        """
        if not self.is_available():
            return self._stub("publish_geotiff", layer_name, geotiff_path)

        coverage_store = coverage_store or f"{layer_name}_store"
        if not os.path.exists(geotiff_path):
            return {"success": False, "error": f"GeoTIFF not found: {geotiff_path}"}

        try:
            # Upload via PUT to coverage store
            url = f"{self.rest_url}/workspaces/{workspace}/coveragestores/{coverage_store}/file.geotiff"
            with open(geotiff_path, "rb") as f:
                r = _REQUESTS.put(
                    url,
                    data=f.read(),
                    auth=self.auth,
                    headers={"Content-Type": "image/tiff"},
                    timeout=120,  # large file upload
                )
            if r.status_code not in (201, 200):
                return {
                    "success": False,
                    "status_code": r.status_code,
                    "response": r.text[:300],
                }

            # GeoServer auto-creates layer dengan nama coverage_store
            wms_url = (
                f"{self.base_url}/{workspace}/wms?"
                f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
                f"&LAYERS={workspace}:{layer_name}&CRS=EPSG:4326"
                f"&BBOX={{minY}},{{minX}},{{maxY}},{{maxX}}"
                f"&WIDTH=512&HEIGHT=512&FORMAT=image/png&TRANSPARENT=true"
            )
            wmts_template = (
                f"{self.base_url}/gwc/service/wmts/rest/{workspace}:{layer_name}/"
                f"EPSG:4326/EPSG:4326:{{z}}/{{y}}/{{x}}.png"
            )
            return {
                "success": True,
                "workspace": workspace,
                "coverage_store": coverage_store,
                "layer_name": layer_name,
                "full_layer_id": f"{workspace}:{layer_name}",
                "wms_url_template": wms_url,
                "wmts_url_template": wmts_template,
                "leaflet_compatible": True,
                "method_monitor": {
                    "selected_method": "GeoServer REST API (GeoTIFF coverage)",
                    "why_chosen": (
                        "GeoServer adalah open-source standar untuk publish geospatial data "
                        "sebagai web service. WMS/WMTS protocol = compatible dengan Leaflet/Mapbox/QGIS/ArcGIS — "
                        "semua client bisa konsumsi tile-nya tanpa modifikasi."
                    ),
                    "why_not_alternatives": [
                        {"alternative": "MapServer", "reason_rejected": "Less mature REST API, harder to script"},
                        {"alternative": "Direct PNG export", "reason_rejected": "Tidak interaktif, tidak ada zoom/pan dynamic"},
                        {"alternative": "Cloud-Optimized GeoTIFF (COG) di S3", "reason_rejected": "Butuh COG-aware client, tidak universal"},
                    ],
                    "limitations": [
                        "Butuh GeoServer instance running (Docker recommended)",
                        "Memory tinggi untuk raster besar",
                        "Auth credential perlu dijaga keamanannya",
                    ],
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================================================================
    # SLD STYLE (untuk flood color ramp)
    # ==================================================================
    def create_flood_probability_style(
        self,
        workspace: str,
        style_name: str = "flood_probability",
    ) -> Dict[str, Any]:
        """
        Create SLD style: flood probability dengan color ramp purple-magenta-cyan.
        Cocok dengan brand Nexlytics.
        """
        if not self.is_available():
            return self._stub("create_flood_probability_style", style_name)

        sld_xml = self._build_flood_probability_sld(style_name)
        try:
            # Step 1: create style metadata
            r1 = _REQUESTS.post(
                f"{self.rest_url}/workspaces/{workspace}/styles",
                json={"style": {"name": style_name, "filename": f"{style_name}.sld"}},
                auth=self.auth,
                timeout=10,
            )
            # Step 2: upload SLD content
            r2 = _REQUESTS.put(
                f"{self.rest_url}/workspaces/{workspace}/styles/{style_name}",
                data=sld_xml,
                headers={"Content-Type": "application/vnd.ogc.sld+xml"},
                auth=self.auth,
                timeout=10,
            )
            return {
                "success": r2.status_code in (200, 201),
                "style_name": style_name,
                "workspace": workspace,
                "sld_size_bytes": len(sld_xml),
                "status": {"create_meta": r1.status_code, "upload_sld": r2.status_code},
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def apply_style_to_layer(
        self,
        workspace: str,
        layer_name: str,
        style_name: str,
    ) -> Dict[str, Any]:
        """Set default style dari layer."""
        if not self.is_available():
            return self._stub("apply_style", layer_name, style_name)
        try:
            r = _REQUESTS.put(
                f"{self.rest_url}/layers/{workspace}:{layer_name}",
                json={"layer": {"defaultStyle": {"name": style_name, "workspace": workspace}}},
                auth=self.auth,
                timeout=10,
            )
            return {
                "success": r.status_code in (200, 201),
                "layer": f"{workspace}:{layer_name}",
                "applied_style": f"{workspace}:{style_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================================================================
    # LAYER MANAGEMENT
    # ==================================================================
    def list_layers(self, workspace: Optional[str] = None) -> Dict[str, Any]:
        """List published layers (optionally filtered by workspace)."""
        if not self.is_available():
            return self._stub("list_layers")
        try:
            url = (
                f"{self.rest_url}/workspaces/{workspace}/layers.json"
                if workspace else f"{self.rest_url}/layers.json"
            )
            r = _REQUESTS.get(url, auth=self.auth, timeout=10)
            if r.status_code == 200:
                data = r.json().get("layers", {})
                layers = data.get("layer", []) if isinstance(data, dict) else []
                return {
                    "success": True,
                    "n_layers": len(layers),
                    "layers": [l.get("name") for l in layers],
                }
            return {"success": False, "status_code": r.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_layer(self, workspace: str, layer_name: str) -> Dict[str, Any]:
        """Delete layer + underlying coverage store."""
        if not self.is_available():
            return self._stub("delete_layer", layer_name)
        try:
            r = _REQUESTS.delete(
                f"{self.rest_url}/layers/{workspace}:{layer_name}.json?recurse=true",
                auth=self.auth,
                timeout=10,
            )
            return {"success": r.status_code in (200, 404), "status_code": r.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================================================================
    # HELPERS
    # ==================================================================
    def _build_flood_probability_sld(self, style_name: str) -> str:
        """SLD XML untuk flood probability color ramp Nexlytics."""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor version="1.0.0"
  xmlns="http://www.opengis.net/sld"
  xmlns:ogc="http://www.opengis.net/ogc"
  xmlns:xlink="http://www.w3.org/1999/xlink"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd">
  <NamedLayer>
    <Name>{style_name}</Name>
    <UserStyle>
      <Title>Flood Probability — Nexlytics</Title>
      <FeatureTypeStyle>
        <Rule>
          <RasterSymbolizer>
            <Opacity>0.8</Opacity>
            <ColorMap type="ramp">
              <ColorMapEntry color="#0A0613" quantity="0.0" opacity="0.0" label="No flood"/>
              <ColorMapEntry color="#3730A3" quantity="0.2" opacity="0.4" label="Low"/>
              <ColorMapEntry color="#6366F1" quantity="0.4" opacity="0.6" label="Moderate-low"/>
              <ColorMapEntry color="#7A42FF" quantity="0.6" opacity="0.7" label="Moderate"/>
              <ColorMapEntry color="#D61CFF" quantity="0.8" opacity="0.85" label="High"/>
              <ColorMapEntry color="#22D3EE" quantity="1.0" opacity="1.0" label="Confirmed flood"/>
            </ColorMap>
          </RasterSymbolizer>
        </Rule>
      </FeatureTypeStyle>
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
"""

    def _stub(self, operation: str, *args) -> Dict[str, Any]:
        return {
            "success": False,
            "operation": operation,
            "warning": "requests library tidak terinstall — operasi di-skip",
            "args": [str(a)[:80] for a in args],
        }


# ==================================================================
# QUICK-USE HELPER
# ==================================================================
def quick_publish_flood_layer(
    geoserver_url: str,
    workspace: str,
    layer_name: str,
    geotiff_path: str,
    style: bool = True,
) -> Dict[str, Any]:
    """
    Convenience: workflow lengkap publish flood mask GeoTIFF ke GeoServer.

    Returns dict dengan WMS URL siap embed di Leaflet.
    """
    client = GeoServerClient(base_url=geoserver_url)

    health = client.health_check()
    if not health.get("available"):
        return {"success": False, "step": "health_check", "details": health}

    # Step 1: workspace
    ws = client.create_workspace(workspace)
    if not ws.get("success") and not ws.get("already_exists"):
        return {"success": False, "step": "create_workspace", "details": ws}

    # Step 2: publish layer
    pub = client.publish_geotiff(workspace, layer_name, geotiff_path)
    if not pub.get("success"):
        return {"success": False, "step": "publish_geotiff", "details": pub}

    result = {"success": True, "publishing": pub}

    # Step 3: optional styling
    if style:
        style_result = client.create_flood_probability_style(workspace)
        if style_result.get("success"):
            apply = client.apply_style_to_layer(workspace, layer_name, "flood_probability")
            result["styling"] = {"created": style_result, "applied": apply}

    result["leaflet_integration"] = {
        "snippet": (
            f"L.tileLayer.wms('{geoserver_url}/{workspace}/wms', {{\n"
            f"  layers: '{workspace}:{layer_name}',\n"
            f"  format: 'image/png',\n"
            f"  transparent: true,\n"
            f"  opacity: 0.7,\n"
            f"}}).addTo(map);"
        ),
    }
    return result
